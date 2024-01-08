import glob
import json
import os

from pydantic import BaseModel, Field
from sklearn.metrics import cohen_kappa_score

from src.config import BASE_DIR
from src.data_handler.models.annotations import Annotation
from src.data_handler.agreement.fleiss import fleissKappa, checkInput
from src.utils.uuid_transformer import convert_to_valid_uuid


class AgreementController(BaseModel):
    annotations: dict[int, list[Annotation]] = Field(default_factory=dict)
    annotation_path: str = "data/annotations/*.json"
    fleiss_kappa: float = 0.0
    cohen_kappa: list[float] = Field(default_factory=list)

    def compute_fleiss_agreement(self):
        print(f"+++ get all annotations from folder: {self.annotation_path} +++")
        self._get_annotations()
        print("+++ create matrix from annotations +++")
        matrix = self._create_matrix()
        print("+++ compute fleiss' kappa +++")
        self.fleiss_kappa = self._use_fleiss_kappa(matrix)

    def _get_annotations(self):
        annotations = []
        annotator = 0
        for filename in glob.glob(os.path.join(BASE_DIR, self.annotation_path)):
            print(f"+++ reading file: {filename} +++")
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

    def _create_matrix(self) -> list[list[int]]:
        counted_annotations = []
        for key, value_list in self.annotations.items():
            for index, obj in enumerate(value_list):
                true = 0
                false = 0
                if obj.is_metaphor:
                    true += 1
                else:
                    false += 1

                if key == 0:
                    counted_annotations.append([true, false])
                else:
                    counted_annotations[index][0] += true
                    counted_annotations[index][1] += false

        return counted_annotations

    def _use_fleiss_kappa(self, matrix: list[list[int]]):
        try:
            checkInput(matrix, len(self.annotations))
        except AssertionError as e:
            print(f"Error in use_fleiss_kappa: {e}")

        return fleissKappa(matrix, len(self.annotations))

    def compute_cohen_agreement(self):
        self._get_annotations()
        annotations_list = self._create_list_from_annotation()
        print("+++ compute cohen's kappa +++")
        kappa_1_2 = cohen_kappa_score(annotations_list[0], annotations_list[1])
        print(f"Agreement between annotator 1 and annotator 2: {kappa_1_2}")
        kappa_1_3 = cohen_kappa_score(annotations_list[0], annotations_list[2])
        print(f"Agreement between annotator 1 and annotator 3: {kappa_1_3}")
        kappa_2_3 = cohen_kappa_score(annotations_list[1], annotations_list[2])
        print(f"Agreement between annotator 2 and annotator 3: {kappa_2_3}")
        self.cohen_kappa = [kappa_1_2, kappa_2_3, kappa_1_3]

    def _create_list_from_annotation(self) -> list[list[int]]:
        annotator_annotations = []
        for key, value_list in self.annotations.items():
            return_list = []
            for obj in value_list:
                if obj.is_metaphor:
                    return_list.append(0)
                else:
                    return_list.append(1)
            annotator_annotations.append(return_list)
        return annotator_annotations

    def get_cohen_examples(self, annotator_a: int, annotator_b: int):
        annotations_a = self.annotations[annotator_a]
        annotations_b = self.annotations[annotator_b]
        agreement = {"is_metaphor": [], "no_metaphor": []}
        disagreement = []
        for index, a in enumerate(annotations_a):
            if a.is_metaphor and annotations_b[index].is_metaphor:
                agreement["is_metaphor"].append(a)
            elif not a.is_metaphor and not annotations_b[index].is_metaphor:
                agreement["no_metaphor"].append(a)
            else:
                disagreement.append(a)

        return agreement, disagreement

    def get_fleiss_examples(self):
        pass

