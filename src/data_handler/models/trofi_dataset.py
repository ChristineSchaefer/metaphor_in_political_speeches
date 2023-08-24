from src.database import Document


class TroFiDataset(Document):
    verb: str
    sentence: str
    verb_idx: int
    label: int

    class Config:
        collection_name = "trofi_dataset"
