# Dependencies
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import gensim

# Python libraries
import json
import subprocess
from collections import Counter

######## Preprocess functions ########
def read_qtl_text(file_path="./QTL_text.json", drop_keys=True, debug=False):
    # Read QTL_text.json file
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

    except FileNotFoundError:
        print("Error: 'QTL_text.json' not found. Please ensure the file exists in the same directory of this program.")
    except Exception as e:
        print(f"An unexpected error occured: {e}")

    # Remove unrelated key-value pairs from the dictionaries ("PMID", "Journal", "Title")
    if drop_keys:
        for d in data:
            d.pop("PMID", None)
            d.pop("Journal", None)
            d.pop("Title", None)
    
    return data


def read_trait_dict(file_path="./Trait_dictionary.txt", strip=True):
    # Read Trait_dictionary.txt file
    try:
        with open(file_path, 'r') as f:
            trait_dictionary = f.read().splitlines()

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found in the project top directory.")
    except Exception as e:
        print(f"An error occurred: {e}")

    # Strip whitespaces
    if strip:
        trait_dictionary = [w.strip() for w in trait_dictionary]

    return trait_dictionary


def load_spacy(download=True, is_uv=False):
    # Download 'en_core_web_sm' for spacy
    if download:
        try:
            if is_uv:
                result = subprocess.run('uv run python -m spacy download en_core_web_sm', shell=True, capture_output=True, text=True, check=True)
            else:
                result = subprocess.run('python -m spacy download en_core_web_sm', shell=True, capture_output=True, text=True, check=True)
            print("Downloading en_core_web_sm (shell output):")
            print(result.stdout)

        except subprocess.CalledProcessError as e:
            print(f"Download 'en_core_web_sm' failed with error: {e}")
            print(f"Stderr: {e.stderr}")

    nlp = spacy.load("en_core_web_sm")

    return nlp


def collect_category(qtl_text, cat_num=1):
    if (type(qtl_text) is not list):
        print("Error in collect_category(): Input should be a list of dict(Abstract: str, Category: int)")
        return

    collected_text = []
    for d in qtl_text:
        if int(d["Category"]) == cat_num:
            collected_text.append(d["Abstract"])
    
    return collected_text


def split_sentence(nlp, docs):
    # Create new list
    new_doc_list = list()
    for doc in nlp.pipe(docs):
        sent_doc = list()
        for sent in doc.sents:
            sent_doc.append(sent.text)
        new_doc_list.append(sent_doc)

    return new_doc_list

def split_token(nlp, docs):
    # Create new list
    new_toc_doc_list = list()
    for sents in docs:
        new_toc_doc = list()
        for s in nlp.pipe(sents):
            for toc in s:
                new_toc_doc.append(toc) # toc.text for str
        new_toc_doc_list.append(new_toc_doc)
    
    return new_toc_doc_list


def preprocess_text(qtl_text, nlp):
    # Collect abstracts with Category 1
    qtl_cat1_text = collect_category(qtl_text, cat_num=1)

    # Split sentences and tokenize (spaCy)
    sent_qtl_cat1_text = split_sentence(nlp, qtl_cat1_text)
    toc_qtl_cat1_text = split_token(nlp, sent_qtl_cat1_text)

    # Covert to both lower case, and lower case and removing stop words
    str_toc_qtl_cat1_text = list()
    lc_toc_qtl_cat1_text = list()
    sw_lc_toc_qtl_cat1_text = list()
    na_sw_lc_toc_qtl_cat1_text = list()
    for sublist in toc_qtl_cat1_text:
        toc = list()
        lc = list()
        sw_lc = list()
        na_sw_lc = list()
        for token in sublist:
            toc.append(token.text)
            token_str = token.text.lower()
            lc.append(token_str)
            if not token.is_stop: # Using spaCy
                sw_lc.append(token_str)
                if token.is_alpha and not token.is_punct:
                    na_sw_lc.append(token.lemma_.lower()) # Incl. lemmatization

        str_toc_qtl_cat1_text.append(toc)
        lc_toc_qtl_cat1_text.append(lc)
        sw_lc_toc_qtl_cat1_text.append(sw_lc)
        na_sw_lc_toc_qtl_cat1_text.append(na_sw_lc)

    return {
        'category_1': qtl_cat1_text,
        'sentence': sent_qtl_cat1_text,
        'token': toc_qtl_cat1_text, # token not str
        'token_str': str_toc_qtl_cat1_text,
        'lower_case': lc_toc_qtl_cat1_text,
        'remove_stop_words': sw_lc_toc_qtl_cat1_text,
        'remove_non_alpha': na_sw_lc_toc_qtl_cat1_text
    }

#### Task 1 functions ########
def compute_word_freq(token_lists):
    flattened_token_list = [item for sublist in token_lists for item in sublist]
    word_freq = Counter(flattened_token_list)

    return word_freq


def compute_tf_idf(token_lists):
    # Calculate tf-idf using sklearn
    sklearn_token_lists = []
    for token_list in token_lists:
        sklearn_token_lists.append(" ".join(token_list))

    vectorizer = TfidfVectorizer()
    tf_idf_matrix = vectorizer.fit_transform(sklearn_token_lists)
    feature_names = vectorizer.get_feature_names_out()

    tf_idf_df = pd.DataFrame(tf_idf_matrix.toarray(), columns=feature_names)
    word_tf_idf_scores = tf_idf_df.sum().to_dict()

    return word_tf_idf_scores


def print_word_cloud_freq(word_freq, file_name="./word_cloud_freq.pdf", format="pdf", is_save=True):
    wc = WordCloud(width=800, height=800, background_color="white").generate_from_frequencies(word_freq)
    plt.figure(figsize=(10, 10))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud from Frequencies")
    if not ((format == "pdf") or (format == "png")):
        format = "pdf"
    if is_save:
        plt.savefig("./word_cloud_freq." + format, format=format, dpi=300)
    plt.show()


def print_word_cloud_tfidf(word_freq, file_name="./word_cloud_tfidf.pdf", format="pdf", is_save=True):
    wc = WordCloud(width=800, height=800, background_color="white").generate_from_frequencies(word_freq)
    plt.figure(figsize=(10, 10))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud from TF-IDF Scores")
    if not ((format == "pdf") or (format == "png")):
        format = "pdf"
    if is_save:
        plt.savefig("./word_cloud_tfidf." + format, format=format, dpi=300)
    plt.show()

#### Task 2 functions ########
def train_word2vec(token_lists, tf_idf_scores):
    # Train Word2Vec model
    model = gensim.models.Word2Vec(token_lists, vector_size=100, window=5, min_count=10, workers=4)

    # Choose top 10 words by TF-IDF scores
    top_ten_words = sorted(tf_idf_scores.items(), key=lambda item: item[1], reverse=True)[:10]
    print(top_ten_words)

    for k, _ in top_ten_words:
        print("Similar 20 words for {}: ".format(k), end="")
        print(model.wv.most_similar(k, topn=20))
        print("\n\n")

#### Task 3 functions ########




if __name__ == '__main__':
    # Read "QTL_text.json" and "Trait_dictionary.txt"
    qtl_text = read_qtl_text()
    trait_dict = read_trait_dict()
    print("Read qtl_text: type: {}, len {}".format(type(qtl_text), len(qtl_text)))

    # Load spacy
    nlp = load_spacy(download=True, is_uv=True) # set download=True for the initial run

    # Preprocessing
    preprocess_results = preprocess_text(qtl_text, nlp)

######## Task 1 ########
    token_lists = preprocess_results["remove_non_alpha"]
    freq = compute_word_freq(token_lists)
    print_word_cloud_freq(freq, format="png", is_save=True)

    tf_idf = compute_tf_idf(token_lists)
    print_word_cloud_tfidf(tf_idf, format="png", is_save=True)

######## Task 2 ########
    train_word2vec(token_lists, tf_idf)

######## Task 3 ########
