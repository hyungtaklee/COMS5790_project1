# Dependencies
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import gensim
from gensim.models.phrases import Phrases, Phraser

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
    """ Return a list of lists of strings
    where each list of strings contains all the sentences from each abstract.
        
    split_sentenct(nlp, docs) -> list(list(str), list(str), ...)
    """
    # Create new list
    new_doc_list = list()
    for doc in nlp.pipe(docs):
        sent_doc = list()
        for sent in doc.sents:
            sent_doc.append(sent.text)
        new_doc_list.append(sent_doc)

    return new_doc_list

def split_token(nlp, docs):
    """ Return a list of lists of strings
    where each list of strings contains all the tokens from each abstract.
        
    split_sentenct(nlp, docs) -> list(list(str), list(str), ...)
    """
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
                # if token.is_alpha and not token.is_punct:
                if (not token.is_punct) and (not token.like_num):
                    na_sw_lc.append(token.lemma_.lower()) # Incl. lemmatization

        str_toc_qtl_cat1_text.append(toc)
        lc_toc_qtl_cat1_text.append(lc)
        sw_lc_toc_qtl_cat1_text.append(sw_lc)
        na_sw_lc_toc_qtl_cat1_text.append(na_sw_lc)

    return {
        'category_1': qtl_cat1_text,
        'doc_list': sent_qtl_cat1_text,
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


def print_word_cloud_freq(word_freq, file_name="word_cloud_freq", format="pdf", is_save=True, is_phrase=False):
    wc = WordCloud(width=800, height=800, background_color="white").generate_from_frequencies(word_freq)
    plt.figure(figsize=(10, 10))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    if is_phrase:
        title_add = " (with phrases)"
    else:
        title_add = ""
    plt.title("Word Cloud from Frequencies" + title_add)
    if not ((format == "pdf") or (format == "png")):
        format = "pdf"
    if is_save:
        plt.savefig(file_name + "." + format, format=format, dpi=300)
    plt.show()


def print_word_cloud_tfidf(word_freq, file_name="word_cloud_tfidf", format="pdf", is_save=True, is_phrase=False):
    wc = WordCloud(width=800, height=800, background_color="white").generate_from_frequencies(word_freq)
    plt.figure(figsize=(10, 10))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    if is_phrase:
        title_add = " (with phrases)"
    else:
        title_add = ""

    plt.title("Word Cloud from TF-IDF Scores" + title_add)
    if not ((format == "pdf") or (format == "png")):
        format = "pdf"
    if is_save:
        plt.savefig(file_name + "." + format, format=format, dpi=300)
    plt.show()


#### Task 2 functions ########
def train_word2vec(token_lists, tf_idf_scores):
    # Train Word2Vec model
    model = gensim.models.Word2Vec(token_lists, vector_size=100, window=5, min_count=10, workers=4)

    # Choose top 10 words by TF-IDF scores
    top_ten_words = sorted(tf_idf_scores.items(), key=lambda item: item[1], reverse=True)[:10]
    print("1. Top 10 words by TF-IDF scores: ", end="")
    print(top_ten_words)
    print("\n\n")

    for k, _ in top_ten_words:
        print("Similar 20 words for {}: ".format(k), end="")
        print(model.wv.most_similar(k, topn=20))
        print("\n")


#### Task 3 functions ########
def phrase_extraction(doc_list, nlp):
    """ Return """
    phrase_token_lists = list()
    phrase_token_lists_space = list()
    phrase_list = list()
    for sents in doc_list:
        tokens = list()
        phrases_space = list()
        phrases = list()
        # Extract tokens
        for sent in nlp.pipe(sents):
            for token in sent:
                if (not token.is_stop) and (not token.is_punct) and (not token.like_num): # Using spaCy
                    # if token.is_alpha and not token.is_punct:
                    tokens.append(token.lemma_.lower()) # Incl. lemmatization

            # Extract phrases
            for chunk in sent.noun_chunks:
                temp = [t.lemma_.lower() for t in chunk
                        if (not t.is_stop) and (not t.is_punct) and (not t.like_num)]

                if len(temp) >= 2:
                    ep = "_".join(temp)
                    phrases.append(ep)
                    eps = " ".join(temp)
                    phrases_space.append(eps)
                    phrase_list.append(eps)

            phrase_token_lists.append(phrases + tokens)
            phrase_token_lists_space.append(phrases_space + tokens)

    return [phrase_token_lists, phrase_token_lists_space, phrase_list]


def phrase_extraction_gensim(token_str_lists):
    # Train a Phrases model
    bigram_model = Phrases(token_str_lists, min_count=1, threshold=1)
    # Export the Phrases model to a Phraser
    bigram_phraser = Phraser(bigram_model)

    transformed_sentences = [bigram_phraser[token_list] for token_list in token_str_lists]

    return phrase_token_lists

if __name__ == '__main__':
    # Read "QTL_text.json" and "Trait_dictionary.txt"
    qtl_text = read_qtl_text()
    trait_dict = read_trait_dict()
    print("Read qtl_text: type: {}, len {}".format(type(qtl_text), len(qtl_text)))

    # Load spacy
    nlp = load_spacy(download=True, is_uv=True) # set download=True for the initial run

    # Preprocessing
    preprocess_results = preprocess_text(qtl_text, nlp)

# ######## Task 1 ########
    token_lists = preprocess_results["remove_non_alpha"]
    freq = compute_word_freq(token_lists)
    print_word_cloud_freq(freq, format="pdf", is_save=True)

    tf_idf = compute_tf_idf(token_lists)
    print_word_cloud_tfidf(tf_idf, format="pdf", is_save=True)

######## Task 2 ########
    print("Word2Vec Result for top 10 words (tokens)")
    train_word2vec(token_lists, tf_idf)

######## Task 3 ########
    [phrase_token_lists, phrase_token_lists_space, phrase_list] = phrase_extraction(preprocess_results["doc_list"], nlp)

    freq_phrase = compute_word_freq(phrase_token_lists)
    print_word_cloud_freq(
        freq_phrase,
        file_name="word_cloud_phrase_freq",
        format="pdf",
        is_save=True,
        is_phrase=True
    )

    tf_idf_phrase = compute_tf_idf(phrase_token_lists_space)
    print_word_cloud_tfidf(
        tf_idf_phrase,
        file_name="./word_cloud_pharase_tfidf",
        format="pdf",
        is_save=True,
        is_phrase=True
    )
    print("Word2Vec Result for top 10 words (tokens + phrases)")
    train_word2vec(phrase_token_lists, tf_idf_phrase)

    # Check the same phrases from trait dict
    cnt = 0
    for p in set(phrase_list):
        if p in trait_dict:
            print("Found {}".format(p))
            cnt = cnt + 1

    print("Total count: {}".format(cnt))
