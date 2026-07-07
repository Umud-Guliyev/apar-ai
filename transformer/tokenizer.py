from transformers import AutoTokenizer

from transformer.config import MODEL_NAME


def create_tokenizer():

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    return tokenizer