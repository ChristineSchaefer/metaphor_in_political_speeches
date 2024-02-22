from pydantic import ConfigDict, Field

from src.database import Document


class Politician(Document):
    """
        Politician Class represents a politician in the database.

        Inherits from:
            Document: Base class for MongoDB documents.
    """
    name: str = Field(..., description="name of the politician")
    party: str | None = Field(default=None, description="name of the party, can be without party")
    model_config = ConfigDict(extra="allow")

    class Settings:
        collection_name = "politicians"

    def __eq__(self, other):
        """
            Overloads the equality operator for Politician objects

            @param other: other object to compare with
            @return true if name of objects are the same else false
        """
        if isinstance(other, Politician):
            return self.name == other.name

