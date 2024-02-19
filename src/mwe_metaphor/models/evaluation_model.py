import numpy as np
from pydantic import BaseModel
from sklearn.metrics import precision_recall_fscore_support


class Evaluate:
    """
        Define evaluation metrics (acc, precision, recall, and f1-score).
        from https://github.com/omidrohanian/metaphor_mwe/blob/master/evaluate.py
    """

    def __init__(self, out, labels):
        """
            Initializes the Evaluate class.
        """
        self.out = np.argmax(out, axis=1).numpy().flatten()
        self.labels = labels.numpy().flatten()

    def accuracy(self):
        """
            Calculates the accuracy of the predictions.

            @returns accuracy of the predictions
        """
        nb_correct = sum(y_t == y_p for y_t, y_p in zip(self.labels, self.out))
        nb_true = len(self.labels)
        score = nb_correct / nb_true
        return score

    def precision_recall_fscore(self, tag_list=[0, 1], average="macro"):
        """
            Calculates the precision, recall and F1-score of the predictions.

            @params tag_list: label values which should be evaluated
            @param average: calculation metrics

            @returns a tuple containing precision, recall and F1
        """
        return precision_recall_fscore_support(self.labels, self.out, average=average, labels=tag_list)[:-1]


class PredictionEvaluationModel(BaseModel):  # serializer
    """
        The PredictionEvaluationModel class is a Pydantic model.
        It is used for defining the structure of the evaluation metrics and for
        validation and serialization of data.
    """
    precision: float
    recall: float
    f1_score: float
    accuracy: float
