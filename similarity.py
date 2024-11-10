import os
import re
import math
import numpy as np
import time

def read_text_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
            return content
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def extract_title_and_text(text_content):
    text_content = text_content.lower()

    # Find title
    title_start = text_content.find("<title>") + len("<title>")
    title_end = text_content.find("</title>")
    title = text_content[title_start:title_end]

    # Find text
    text_start = text_content.find("<text>") + len("<text>")
    text_end = text_content.find("</text>")
    text_part = text_content[text_start:text_end]

    final_string = title + " " + text_part
    final_string = ' '.join(final_string.split())
    return final_string

def remove_punc(input_string):
    pattern = r'[^\w\s_]+'
    substrings = re.split(pattern, input_string)
    substrings = [substring for substring in substrings if substring]
    joined_string = ' '.join(substrings)
    return joined_string

def tokenization(files):
    unique_word_ids = {} 
    for i in files:
        contentOfFile = read_text_file(os.path.join(directory, i))
        extractedPart = extract_title_and_text(contentOfFile)
        normalisedPart = remove_punc(extractedPart)

        current_id = 1
        for word in normalisedPart.split():
            if word not in unique_word_ids:
                unique_word_ids[word] = current_id
                current_id += 1

    return unique_word_ids

def file_tokens(normalisedPart):
    token_count = {}

    for j in normalisedPart:
        if j in token_count:
            token_count[j] += 1 
        else:
            token_count[j] = 1    
    termfrq = {token:value/len(normalisedPart) for token,value in token_count.items()}
    return termfrq

def doc_freq(list1):
    unique_words = set(list1)
    unique_word_list = list(unique_words)
    global dict
    for j in unique_word_list:
        if j in dict:
            dict[j] += 1 
        else:
            dict[j] = 1 
    return dict

def contentTostore(files):
    file_contents = {}
    for i in files:
        contentOfFile = read_text_file(os.path.join(directory, i))
        extractedPart = extract_title_and_text(contentOfFile)
        normalisedPart = remove_punc(extractedPart).split()
        file_contents[i] = file_tokens(normalisedPart)
        doc_freq(normalisedPart)
    return file_contents    

def compute_idf(dict, files):
    idf_dict = {}
    for i in dict:
        if i not in idf_dict:
            idf_dict[i] = math.log(len(files)/dict[i])

    return idf_dict

def compute_cos(tf_dict, idf_dict):
    tf_idf_cross_product = {}
    for doc, term_freq in tf_dict.items():
        tf_idf_cross_product[doc] = {}
        for term, tf in term_freq.items():
            if term in idf_dict:
                tf_idf_cross_product[doc][term] = tf * idf_dict[term]
    return tf_idf_cross_product  

def compute_magnitude(vector):
    return math.sqrt(sum(value ** 2 for value in vector.values()))

def normalize_vector(vector):
    magnitude = compute_magnitude(vector)
    return {term: value / magnitude for term, value in vector.items()}

def compute_cosine_normalized_vectors(tf_idf_cross_product):
    normalized_vectors = {}
    for doc, tf_idf_vector in tf_idf_cross_product.items():
        normalized_vectors[doc] = normalize_vector(tf_idf_vector)
    return normalized_vectors

def compute_dot_product(vector1, vector2):
    return sum(value1 * vector2.get(term, 0) for term, value1 in vector1.items())

def compute_cosine_similarity(dict_a, dict_b):
    common_keys = set(dict_a.keys()) & set(dict_b.keys())
    
    dot_product = sum(dict_a[dim] * dict_b[dim] for dim in common_keys)
    
    norm_a = np.linalg.norm(list(dict_a.values()))
    norm_b = np.linalg.norm(list(dict_b.values()))
    
    similarity = dot_product / (norm_a * norm_b)
    return similarity

def compute_top_k_cosine_similarities(vectors, k=50):
    similarities = []
    doc_names = list(vectors.keys())

    for i in range(len(doc_names)):
        for j in range(i+1, len(doc_names)):
            similarity = compute_cosine_similarity(vectors[doc_names[i]], vectors[doc_names[j]])
            similarities.append(((doc_names[i], doc_names[j]), similarity))
    similarities.sort(key=lambda x: x[1], reverse=True)

    return similarities[:k]

# Main function
if __name__ == "__main__":
    start_time = time.time()

    directory = "C:/Users/Lenovo/Documents/SEIR/25"
    allFiles = os.listdir(directory)

    dict = {}

    tfs = contentTostore(allFiles)

    idfs = compute_idf(dict, allFiles)

    tf_idf_cross_product = compute_cos(tfs, idfs)

    cosine_normalized_vectors = compute_cosine_normalized_vectors(tf_idf_cross_product)

    top_50_cosine_similarities = compute_top_k_cosine_similarities(cosine_normalized_vectors, 50)

    for doc_pair, similarity in top_50_cosine_similarities:
        print(f"The cosine similarity between {doc_pair[0]} and {doc_pair[1]} is {similarity}.")

    end_time = time.time()

    elapsed_time = end_time - start_time

    print("Elapsed time:", elapsed_time, "seconds")