from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

from preprocessing import clean_text
from config import (
    FEATURE_SET,
    WORD_NGRAM,
    CHAR_NGRAM,
)


def build_vectorizer():

    if FEATURE_SET == "word":
        return TfidfVectorizer(
            preprocessor=clean_text,
            analyzer="word",
            ngram_range=WORD_NGRAM,
        )

    if FEATURE_SET == "char":
        return TfidfVectorizer(
            preprocessor=clean_text,
            analyzer="char_wb",
            ngram_range=CHAR_NGRAM,
        )

    if FEATURE_SET == "word_char":

        word = TfidfVectorizer(
            preprocessor=clean_text,
            analyzer="word",
            ngram_range=WORD_NGRAM,
        )

        char = TfidfVectorizer(
            preprocessor=clean_text,
            analyzer="char_wb",
            ngram_range=CHAR_NGRAM,
        )

        return FeatureUnion([
            ("word", word),
            ("char", char),
        ])

    raise ValueError("Unknown FEATURE_SET")