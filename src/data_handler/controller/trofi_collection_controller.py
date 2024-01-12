from pydantic import BaseModel, Field

from src.data_handler.models.annotations import Annotation
from src.data_handler.models.trofi_dataset import TroFiDataset


class TrofiCollectionController(BaseModel):
    annotations: list[Annotation]
    trofi: list[TroFiDataset] | None = Field(default_factory=list)

    def create_trofi_object_and_save_in_collection(self):
        for annotation in self.annotations:
            trofi_object = TroFiDataset(
                verb=annotation.lexem,
                sentence=annotation.sentence,
                verb_idx=annotation.index_verb,
                label=1 if annotation.is_metaphor else 0
            ).save()
            self.trofi.append(trofi_object)

        print(self.trofi)
