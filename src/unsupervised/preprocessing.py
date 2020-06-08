# Code for preprocessing

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction import text
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk import word_tokenize
from pprint import pprint
import pandas as pd
import altair as alt
import numpy as np
import nltk
import re


def prepare_data(csv_url):
    """
    Fetches data and gets rid of any duplicated text
    """
    # read in csv
    df = pd.read_csv(csv_url)
    # filter to relevant section
    df = df[df.column_name == "Initial / Intake Allegation"]
    # filter to relevant columns
    df = df[["cr_id", "text"]]
    print("There are {} complaints".format(df.shape[0]))
    # drop allegations with same id + text
    df = df.drop_duplicates(["cr_id", "text"])
    print("There are {} unique complaints".format(df.shape[0]))

    # get rid of repeated text
    df["DuplicateText"] = df["text"].apply(split_half_and_compare)
    df["text"] = df.apply(
        lambda row: row["text"]
        if row["DuplicateText"] == False
        else row["text"][0 : len(row["text"]) // 2],
        axis=1,
    )
    df = df[["cr_id", "text"]]
    print(df.shape)
    return df


def cosine_sim_vectors(vec1, vec2):
    """
    Compute cosine similarity of two vectors
    """
    vec1 = vec1.reshape(1, -1)
    vec2 = vec2.reshape(1, -1)
    return cosine_similarity(vec1, vec2)[0][0]


def split_half_and_compare(str_input):
    """
    Split vectors to check if allegation text is repeated
    """
    if (
        len(str_input) < 2
        or str_input in ["nfi", "NFI"]
        or str_input in nltk.corpus.stopwords.words("english")
    ):
        return False
    first_half = str_input[0 : len(str_input) // 2]
    second_half = str_input[
        len(str_input) // 2
        if len(str_input) % 2 == 0
        else ((len(str_input) // 2) + 1) :
    ]

    vectorizer = CountVectorizer().fit_transform([first_half, second_half])
    vectors = vectorizer.toarray()
    similarity_score = cosine_sim_vectors(vectors[0], vectors[1])
    return similarity_score > 0.9


class PreProcess:

    """
    Class for pre-processing a csv of text documents into a sparse matrix of
    counts following scikit-learn's CountVectorizer

    Sources:
        - sklearn: https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/feature_extraction/text.py
        - tokenizers: https://kavita-ganesan.com/how-to-use-countvectorizer/#.XtggmMa23yU

    """

    def __init__(self, raw_data, additional_stopwords=set(), stem=True):
        self.raw_data = raw_data
        self.stemmer = PorterStemmer()
        self.stop_words = set(nltk.corpus.stopwords.words("english")).union(
            additional_stopwords
        )
        self.stem = stem
        self.vectorizer = None
        self.doc_term_matrix = None

    def _tokenize_text(self, text):
        """
        Strips punctuation, and everything that isn't alphabetic, tokenizes.
        Stems by default.
        """
        tokenized_text = []

        # drop whatever isn't a word with letters or an apostrophe
        for token in word_tokenize(text):
            # to lowercase
            token = token.lower()
            # substitute whatever is not alphabetic
            token = re.sub("[^a-z]", "", token)
            if token:
                if token not in self.stop_words:
                    if self.stem:
                        # stemming
                        tokenized_text.append(self.stemmer.stem(token))
                    else:
                        tokenized_text.append(token)

        return tokenized_text

    def _vectorize(self):
        """
        Launch a vectorizer with CountVectorizer
        """
        # instantiate vectorizer w/ our custom analyzer
        # by default we drop words that appear in more than 80% of documents and
        # that don't appear in more than one document
        # we override the analyzer with our tokenizer method
        self.vectorizer = CountVectorizer(
            max_df=0.8, min_df=10, analyzer=self._tokenize_text
        )

    def _fit_vectorizer(self):
        """
        Fit vectorizer and create a doc_term_matrix
        """
        # launch the CountVectorizer object
        self._vectorize()
        # fit it
        self.doc_term_matrix = self.vectorizer.fit_transform(
            self.raw_data.text.values.astype("U")
        )
