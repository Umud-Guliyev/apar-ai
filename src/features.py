from sklearn.feature_extraction.text import TfidfVectorizer

from config import NGRAM_RANGE, MIN_DF, MAX_DF
from preprocessing import clean_text


def build_vectorizer():
    return TfidfVectorizer(
        preprocessor=clean_text,
        ngram_range=NGRAM_RANGE,
        min_df=MIN_DF,
        max_df=MAX_DF,
    )