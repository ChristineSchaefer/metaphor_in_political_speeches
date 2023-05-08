from pydantic import Extra, Field

from src.database import Document


class Politician(Document):
    name: str
    party: str | None = Field(default_factory=None)

    class Config:
        extra = Extra.allow
        collection_name = "politicians"

    def __eq__(self, other):
        if isinstance(other, Politician):
            return self.name == other.name

