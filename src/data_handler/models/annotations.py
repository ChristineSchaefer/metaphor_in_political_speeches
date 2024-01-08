from pydantic import Field, Extra

from src.database import Document


class Annotation(Document):
    speech_id: str
    sentence: str
    index_sentence: int = 0
    verb: str
    lexem: str = ""
    index_verb: int = 0
    basic_meaning: list = Field(default_factory=list)
    contextual_meaning: list = Field(default_factory=list)
    is_metaphor: bool = False

    class Config:
        collection_name = "annotations"
        extra = Extra.allow

    def __hash__(self) -> int:
        return self.speech_id.__hash__()  # or self.id.__hash__()

