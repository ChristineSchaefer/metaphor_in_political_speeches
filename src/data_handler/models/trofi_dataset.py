from src.database import Document


class TroFiDataset(Document):
    verb: str
    sentence: str
    verb_idx: int
    label: int

    class Config:
        collection_name = "trofi_dataset"

    def __hash__(self) -> int:
        return self.id.__hash__()  # or self.id.__hash__()
