from pymongo.collection import Collection


def save_many(collection: Collection, documents: list):
    collection.insert_many(documents)
