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
    """
        Controller class for managing and calculating agreement scores based on annotations.
        This class is responsible for handling annotations and producing agreement metrics including Fleiss Kappa and Cohen Kappa.
    """
    annotations: dict[int, list[Annotation]] = Field(default_factory=dict, description="dictionary with annotators annotations")
    annotation_path: str = Field(default="data/annotations/*.json", description="path to annotation files")
    fleiss_kappa: float = Field(default=0.0, description="fleiss kappa value")
    cohen_kappa: list[float] = Field(default_factory=list, description="cohen kappa values for different annotator pairs")

    def compute_fleiss_agreement(self):
        """
            Function to compute and return the Fleiss' Kappa agreement measure for the annotations.
        """
        print(f"+++ get all annotations from folder: {self.annotation_path} +++")
        self._get_annotations()
        print("+++ create matrix from annotations +++")
        matrix = self._create_matrix()
        print("+++ compute fleiss' kappa +++")
        self.fleiss_kappa = self._use_fleiss_kappa(matrix)

    def _get_annotations(self):
        """
            Private function to fetch and set all available annotations.
        """
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
        """
            Private function responsible for transforming the fetched annotations into a suitable matrix format.

            @return matrix for annotations
        """
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
        """
            Private method for utilising the Fleiss' Kappa methodology to compute the agreement score based on the created matrix.

            @param matrix: matrix of annotations
        """
        try:
            checkInput(matrix, len(self.annotations))
        except AssertionError as e:
            print(f"Error in use_fleiss_kappa: {e}")

        return fleissKappa(matrix, len(self.annotations))

    def compute_cohen_agreement(self):
        """
            Function to compute and set the Cohen's Kappa agreement measure for the annotations.
        """
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
        """
            Private method to transform the annotations into a list, used for Cohen's Kappa computation.

            @return list with list of annotations
        """
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
        """
            Function to get examples of annotator pair.

            @param annotator_a: annotator a
            @param annotator_b: annotator b
        """
        annotations_a = self.annotations[annotator_a]
        annotations_b = self.annotations[annotator_b]
        agreement = {"is_metaphor": set(), "no_metaphor": set()}
        disagreement = set()
        for index, a in enumerate(annotations_a):
            if a.is_metaphor and annotations_b[index].is_metaphor:
                agreement["is_metaphor"].add(a)
            elif not a.is_metaphor and not annotations_b[index].is_metaphor:
                agreement["no_metaphor"].add(a)
            else:
                disagreement.add(a)

        return agreement, disagreement

    def update_annotation_collection(self):
        """
            Function to update the annotation collection based on annotator agreement
        """
        matrix = self._create_matrix()
        for index, pair in enumerate(matrix):
            annotation = self.annotations[0][index]
            if pair[0] > pair[1]:
                annotation.is_metaphor = True
            else:
                annotation.is_metaphor = False
            annotation.save()

