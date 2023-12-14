import glob
import json
import os
from collections import Counter

from pydantic import BaseModel, Field

from src.config import BASE_DIR
from src.data_handler.models.annotations import Annotation
from src.utils.uuid_transformer import convert_to_valid_uuid


class AgreementController(BaseModel):
    annotations: dict[int, list[Annotation]] = Field(default_factory=dict)
    annotation_path: str = "data/annotations/*.json"

    # TODO read json data into annotation list for each annotator - check -
    #  iterate over annotation with same id
    #  count if is_metaphor
    #  create matrix with annotation_id and categories (nparray 252x2, values between 0-3)

    def get_annotations(self):
        annotations = []
        annotator = 0
        for filename in glob.glob(os.path.join(BASE_DIR, self.annotation_path)):
            with open(filename, encoding='utf-8', mode='r') as currentFile:
                file = currentFile.read().replace('\n', '')
                data = json.loads(file)
                for i in data:
                    i["_id"] = convert_to_valid_uuid(i["_id"]["$binary"]["base64"])
                    annotation = Annotation(**i)
                    annotations.append(annotation)
                self.annotations[annotator] = annotations
                annotator += 1
                annotations = []

    def count_metaphor_annotation(self):
        # TODO
        annotation_list = []
        for annotation in self.annotations.values():
            pass
        metaphor_counts = Counter(a.is_metaphor for a in annotation_list)

        return metaphor_counts[True], metaphor_counts[False]


if __name__ == '__main__':
    a = AgreementController()
    a.get_annotations()
