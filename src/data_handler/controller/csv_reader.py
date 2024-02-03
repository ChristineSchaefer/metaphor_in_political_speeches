from pydantic import BaseModel, Field
import duden

from src.data_handler.models.annotations import Annotation
from src.utils.text_handler import read_from_csv_and_write_to_dict, normalize


class CSVController(BaseModel):
    """
        Controller class for managing and handling CSV files.
        This class is responsible for reading a CSV file and transforming its content to the required format.
    """
    path: str = Field(..., description="path to csv")
    annotation_input: list = Field(default_factory=list, description="list with annotation dicts")
    annotations: list[Annotation] = Field(default_factory=list, description="list with annotation objects")

    def create_annotation_object(self):
        """
            Function to create and return an annotation object based on the CSV content.
            Also, It puts required information like basic meaning of sentence in it.
        """
        self.annotation_input = read_from_csv_and_write_to_dict(self.path)
        for annotation in self.annotation_input[1:]:
            self.annotations.append(Annotation(
                basic_meaning=self._find_basic_meaning(annotation.get("lexem")),
                index_verb=self._get_index_of_verb_in_sentence(annotation.get("sentence"), annotation.get("verb")),
                index_sentence=int(annotation.get("sentence_index"))-1,
                sentence=annotation.get("sentence"),
                speech_id=annotation.get("speech_id"),
                verb=annotation.get("verb"),
                lexem=annotation.get("lexem", "")).save())
        print(self.annotations)

    @staticmethod
    def _find_basic_meaning(word: str) -> list:
        """
            A private function to derive and return the basic meaning of a sentence from the CSV content.
            It uses the duden API.

            @param word: word for which basic meaning is sought

            @return list with basic meaning
        """
        try:
            meaning = duden.get(word).meaning_overview
            if type(meaning) != list:
                return [meaning]
            return meaning
        except AttributeError:
            return []

    @staticmethod
    def _get_index_of_verb_in_sentence(sentence: str, word: str) -> int:
        """
            A private function to find and return the index of the verb in a sentence from the CSV content.

            @param sentence: full sentence
            @param word: word for which the index is searched

            @return index of word in sentence
        """
        normalized_sentence = normalize(sentence)
        return normalized_sentence.split().index(word)
