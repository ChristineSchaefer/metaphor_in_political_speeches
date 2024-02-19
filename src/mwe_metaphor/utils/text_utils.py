from src.config import Settings
from src.mwe_metaphor.utils.tsvlib import iter_tsv_sentences, TSVSentence


def tokenize(sentences, language_model) -> list:
    tokenized_texts = []
    for sent in sentences:
        tokenized_sent = []
        doc = language_model(sent)
        for token in doc:
            if not token.text.isspace():
                tokenized_sent.append(token.text.lower())
        tokenized_texts.append(tokenized_sent)
    return tokenized_texts


def load_data(settings: Settings, path: str) -> list[TSVSentence]:
    """
        A method that loads data for training and validation.
        It sets the data as attributes to the instance for later usage.

        @param path: path to the mwe korpus
        @param settings: project settings

        @returns list with TSVSentence objects
    """
    with open(f"{settings.mwe_dir}/{path}") as f:
        return list(iter_tsv_sentences(f))
