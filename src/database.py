import uuid
from typing import Mapping, Any, Self

from pydantic import ConfigDict, BaseModel, Field
from pymongo import MongoClient
from pymongo.collection import Collection

from src.config import get_settings


env = get_settings()
client = MongoClient(env.mdb_connection_string(), uuidRepresentation="standard")
metaphor_identification_db = client[env.db_name]


class Document(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, description="id of document")
    model_config = ConfigDict(extra="forbid")

    class Settings:
        collection_name = ""

    def __init__(self, **kwargs):
        """
            Initialize a document.
            Convert mongodb specific field _id to id.
        """
        if kwargs.get("_id"):
            kwargs["id"] = kwargs.pop("_id")
        super().__init__(**kwargs)

    @classmethod
    def _get_collection_name(cls, ) -> str:
        """
            @returns collection name
        """
        return cls.Config.collection_name

    @classmethod
    def _get_collection_from_name(cls, name: str) -> Collection:
        """
            @returns collection from name
        """
        return metaphor_identification_db[name]

    @classmethod
    def collection(cls, ) -> Collection:
        """
            @returns collection
        """
        name = cls._get_collection_name()
        if not name:
            raise Exception(f"{cls}.Config.collection_name is missing!")
        return cls._get_collection_from_name(name)

    def save(self):
        """
            Inserts or updates a document with this id in the database.
        """
        self.collection().replace_one({"_id": self.id}, self.dict(exclude={"id"}), upsert=True)
        return self

    @classmethod
    def find(cls, filter: Mapping[str, Any] = None, *args, **kwargs) -> Self:
        """
        Shortcut method that retrieves all document and instantiates the models for them.
        """
        if filter is None:
            filter = {}
        return [cls(**doc) for doc in cls.collection().find(filter, *args, **kwargs)]