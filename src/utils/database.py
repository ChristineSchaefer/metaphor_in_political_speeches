import glob
import json
import os
from pathlib import Path

from bson import ObjectId
from pymongo.collection import Collection

from src.config import BASE_DIR
from src.database import metaphor_identification_db
from src.utils.text_handler import make_json_from_csv
from src.utils.uuid_transformer import oid_to_uuid, bson_to_uuid


def save_many(collection: Collection, documents: list):
    """
        Helper method for saving multiple documents.

        @param collection: mongodb collection
        @param documents: list with documents to save
    """
    collection.insert_many(documents)


def create_collection_from_json():
    """
        Reads JSON files from a specified directory, transforms the data if necessary,
        and saves each file as a MongoDB collection.
    """
    for file in glob.glob(os.path.join(BASE_DIR, "data/collections/*.json")):
        collection_name = Path(file).stem
        collection = metaphor_identification_db[collection_name]
        with open(file) as f:
            file_data = json.load(f)
            for document in file_data:
                if "$oid" in document["_id"]:
                    uuid = oid_to_uuid(ObjectId(document["_id"]["$oid"]))
                    document.pop("_id")
                    document["_id"] = uuid
                elif "$binary" in document["_id"]:
                    uuid = bson_to_uuid(document["_id"])
                    document.pop("_id")
                    document["_id"] = uuid
                if document.get("speaker", None) is not None:
                    if "$binary" in document["speaker"]["id"]:
                        uuid = bson_to_uuid(document["speaker"]["id"])
                        document["speaker"].pop("id")
                        document["speaker"]["id"] = uuid
            collection.insert_many(file_data)


def create_collection_from_csv():
    """
        Reads CSV files from a specified directory, transforms the data into JSON format,
        and saves each file as a MongoDB collection.
    """
    for file in glob.glob(os.path.join(BASE_DIR, "data/collections/*.csv")):
        collection_name = Path(file).stem
        collection = metaphor_identification_db[collection_name]
        data = make_json_from_csv(file)
        for document in data:
            if "$oid" in document["_id"]:
                uuid = oid_to_uuid(ObjectId(document["_id"]["$oid"]))
                document.pop("_id")
                document["_id"] = uuid
        collection.insert_many(data)


def init_db():
    """
        Initializes the database by creating collections from the CSV and JSON files in a specified directory.
    """
    create_collection_from_csv()
    create_collection_from_json()
