from pydantic import Extra

from src.database import Document


class Politician(Document):
    name: str
    party: str

    class Config:
        extra = Extra.allow
        collection_name = "politicians"

    def __eq__(self, other):
        if isinstance(other, Politician):
            return self.name == other.name

