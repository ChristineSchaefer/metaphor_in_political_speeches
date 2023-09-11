import spacy
from pydantic import BaseModel
from spacy_conll import init_parser, ConllParser


class SpacyModel(BaseModel):
    language_model: str
    language: str = ""

    def get_language_model(self):
        return spacy.load(self.language_model)


class CoNLLParserModel(SpacyModel):
    def init_parser(self):
        return init_parser(self.language_model, "spacy", include_headers=True)

    def get_parser(self):
        return ConllParser(self.init_parser())
