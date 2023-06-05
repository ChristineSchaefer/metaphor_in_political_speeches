from pydantic import Extra

from src.data_handler.models.politician import Politician
from src.database import Document


class Speech(Document):
    text: str
    speaker: Politician
    url: str = ""

    class Config:
        collection_name = "speeches_raw"
        extra = Extra.allow
