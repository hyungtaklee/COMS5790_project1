# Dependencies
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import spacy

# Python libraries
import json
import subprocess

######## Preprocess 
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
    for doc in nlp.pipe(docs):
        for token in doc:
            print(token)
            break


def preprocess_text(qtl_text, nlp):
    # Collect abstracts with Category 1
    qtl_text = collect_category(qtl_text, cat_num=1)

    # Split sentences and tokenize (spaCy)

    # Convert to lower case (Python string)

    # Remove stop words (spaCy)


    return qtl_text


def compute_word_freq(trait_dict, docs):
    pass


def compute_tf_idf(trait_dict, docs):
    pass


# Word Cloud part
# wc = WordCloud(background_color="white", repeat=True, mask=mask)
# wc.generate(text)

# plt.axis("off")
# plt.imshow(wc, interpolation="bilinear")
# plt.show()

# Task 2


# Task 3


if __name__ == '__main__':
    # Read "QTL_text.json" and "Trait_dictionary.txt"
    qtl_text = read_qtl_text()
    trait_dict = read_trait_dict()
    print("Read qtl_text: type: {}, len {}".format(type(qtl_text), len(qtl_text)))

    # Load spacy
    nlp = load_spacy(download=True, is_uv=True)

    # Collect only abstracts of category 1
    c = collect_category(qtl_text, cat_num=1)
    print("After collecing abstract in Category 1: len(Category) = {}, Example: {}".format(len(c), c[4]))

    split_sentence(nlp, c)

