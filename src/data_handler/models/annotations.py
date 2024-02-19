from pydantic import ConfigDict, Field

from src.database import Document


class Annotation(Document):
    """
        Annotation class extends Document and represents an annotation model in the database.
    """
    speech_id: str = Field(..., description="id of the speech")
    sentence: str = Field(..., description="sentence used for the annotation")
    index_sentence: int = Field(default=0, description="index of sentence in speech")
    verb: str = Field(..., description="analyzed verb")
    lexem: str = Field(default="", description="lexem of verb")
    index_verb: int = Field(default=0, description="index of verb in sentence")
    basic_meaning: list = Field(default_factory=list, description="basic meaning of verb based on duden")
    contextual_meaning: list = Field(default_factory=list, description="contextual meaning of verb in sentence")
    is_metaphor: bool = Field(default=False, description="metaphorical use of verb")
    model_config = ConfigDict(extra="allow")

    class Settings:
        collection_name = "annotations"

    def __hash__(self) -> int:
        """
            Override the default hash behavior to set the hash value of an Annotation object
            to be the hash value of its associated speech id.
            It ensures that each annotation has a unique hash value.
        """
        return self.speech_id.__hash__()  # or self.id.__hash__()

