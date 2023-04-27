from src.database import Document


class Politician(Document):
    name: str
    party: str

    class Config:
        collection_name = "politicians"

