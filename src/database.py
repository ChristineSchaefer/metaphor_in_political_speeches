import json
import uuid

from pydantic import BaseModel, Extra, Field
from pymongo.collection import Collection

from src.utils.database import get_database


class Document(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)

    class Config:
        collection_name = ""
        extra = Extra.forbid

    @classmethod
    def _get_collection_name(cls, ) -> str:
        return cls.Config.collection_name

    @classmethod
    def _get_collection_from_name(cls, name: str) -> Collection:
        return get_database()[name]

    @classmethod
    def collection(cls, ) -> Collection:
        name = cls._get_collection_name()
        if not name:
            raise Exception(f"{cls}.Config.collection_name is missing!")
        return cls._get_collection_from_name(name)

    def savable_dict(self) -> dict:
        savable = json.loads(self.json())
        savable.pop("id")
        return savable

    # TODO replace_one with id does not work yet
    def save(self):
        """Inserts or updates a document with this id in the database"""
        filter = {"_id": str(self.id)}
        self.collection().replace_one(filter, self.savable_dict(), upsert=True)
        return self