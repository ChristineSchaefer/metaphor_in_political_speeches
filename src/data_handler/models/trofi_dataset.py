from pydantic import ConfigDict, Field

from src.database import Document


class TroFiDataset(Document):
    """
        TroFi data Class represents a politician in the database.

        Inherits from:
            Document: Base class for MongoDB documents.
    """
    verb: str = Field(..., description="potential metaphoric verb")
    sentence: str = Field(..., description="sentence of speech")
    verb_idx: int = Field(..., description="index of potential metaphoric verb")
    label: int = Field(..., description="metaphorical label")

    class Settings:
        collection_name = "trofi_dataset"

    def __hash__(self) -> int:
        """
            Computes a unique hash value for a TroFiDataset instance.

            This method is used for hashing a TroFiDataset instance, using its 'id'
            attribute to generate a unique hash value. This facilitates the use of
            TroFiDataset objects in hashed collections (e.g. sets, dicts).

            @return hash for object
        """
        return self.id.__hash__()  # or self.id.__hash__()
