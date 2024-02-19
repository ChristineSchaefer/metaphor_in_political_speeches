from pydantic import ConfigDict, Field

from src.data_handler.models.politician import Politician
from src.database import Document


class Speech(Document):
    """
        Speech Class represents a speech in the database.

        Inherits from:
            Document: Base class for MongoDB documents.
    """
    text: str = Field(..., description="text of speech")
    speaker: Politician = Field(..., description="speaker of the speech")
    url: str = Field(default="", description="url of the speech")
    model_config = ConfigDict(extra="allow")

    class Settings:
        collection_name = "speeches_raw"
