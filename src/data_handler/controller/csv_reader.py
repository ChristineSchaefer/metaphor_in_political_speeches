from pydantic import BaseModel, Field
import duden

from src.data_handler.models.annotations import Annotation
from src.utils.text_handler import read_from_csv_and_write_to_dict, normalize


class CSVController(BaseModel):
    path: str
    annotation_input: list = Field(default_factory=list)
    annotations: list[Annotation] = Field(default_factory=list)

    def create_annotation_object(self):
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
        try:
            meaning = duden.get(word).meaning_overview
            if type(meaning) != list:
                return [meaning]
            return meaning
        except AttributeError:
            return []

    @staticmethod
    def _get_index_of_verb_in_sentence(sentence: str, word: str) -> int:
        normalized_sentence = normalize(sentence)
        return normalized_sentence.split().index(word)
