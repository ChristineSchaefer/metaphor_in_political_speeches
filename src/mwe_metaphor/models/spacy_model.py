import spacy
from pydantic import BaseModel, Field
from spacy_conll import init_parser, ConllParser


class SpacyModel(BaseModel):
    """
        The SpacyModel class defines a spaCy language model.
    """
    language_model: str = Field(..., description="name of language model")
    language: str = Field(default="", description="language")

    def get_language_model(self):
        """
        Loads and returns the spaCy language model.

        @returns the loaded language model
        """
        return spacy.load(self.language_model)


class CoNLLParserModel(SpacyModel):
    """
        The CoNLLParserModel class extends the SpacyModel to include spaCy-CoNLL parser.
        CoNLLParserModel is used to generate Conll-U formatted parses from the text data
        using the specified spaCy model.
    """
    def init_parser(self):
        """
            Initializes the spaCy parser with the CoNLL configuration.

            @returns initialized writer for CoNLL-U TokenList
        """
        return init_parser(self.language_model, "spacy", include_headers=True)

    def get_parser(self):
        """
            Returns a spaCy-CoNLL parser.

            @returns the CoNLL parser for the model
        """
        return ConllParser(self.init_parser())
