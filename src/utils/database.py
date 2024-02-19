from pymongo.collection import Collection


def save_many(collection: Collection, documents: list):
    """
        Helper method for saving multiple documents.

        @param collection: mongodb collection
        @param documents: list with documents to save
    """
    collection.insert_many(documents)
