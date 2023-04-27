from pymongo import MongoClient

from src.config import get_settings


def get_database():
    env = get_settings()
    client = MongoClient(env.mdb_connection_string(), uuidRepresentation="standard")
    return client[env.db_name]
