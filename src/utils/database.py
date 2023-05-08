from pymongo import MongoClient
from pymongo.collection import Collection

from src.config import get_settings


def get_database():
    env = get_settings()
    client = MongoClient(env.mdb_connection_string(), uuidRepresentation="standard")
    return client[env.db_name]


def save_many(collection: Collection, documents: list):
    collection.insert_many(documents)
