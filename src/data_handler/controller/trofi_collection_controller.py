from pydantic import BaseModel, Field

from src.data_handler.models.annotations import Annotation
from src.data_handler.models.trofi_dataset import TroFiDataset


class TrofiCollectionController(BaseModel):
    """
        Controller class for managing and manipulating TroFi dataset collection.
        The class is responsible for creating TroFi objects from the annotations and saving them into the collection.
    """
    annotations: list[Annotation] = Field(..., description="list with annotation objects")
    trofi: list[TroFiDataset] | None = Field(default_factory=list, description="list with trofi objects")

    def create_trofi_object_and_save_in_collection(self):
        """
            Method to create TroFi objects based on the annotations and save them into the TroFi dataset collection.
            This involves creating TroFiDataset instances with properties derived from the annotations,
            saving these instances, and appending them to the TroFi collection.
        """
        for annotation in self.annotations:
            trofi_object = TroFiDataset(
                verb=annotation.lexem,
                sentence=annotation.sentence,
                verb_idx=annotation.index_verb,
                label=1 if annotation.is_metaphor else 0
            ).save()
            self.trofi.append(trofi_object)

        print(self.trofi)
