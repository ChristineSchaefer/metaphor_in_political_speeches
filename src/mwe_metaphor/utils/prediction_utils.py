import numpy as np

from src.mwe_metaphor.models.evaluation_model import PredictionEvaluationModel


def compute_average(evaluation_results: list[PredictionEvaluationModel]):
    precision_scores = [value.precision for value in evaluation_results]
    average_precision = np.mean(precision_scores)

    recall_scores = [value.recall for value in evaluation_results]
    average_recall = np.mean(recall_scores)

    f1_scores = [value.f1_score for value in evaluation_results]
    average_f1_score = np.mean(f1_scores)

    accuracy_score = [value.accuracy for value in evaluation_results]
    average_accuracy = np.mean(accuracy_score)

    return average_accuracy, average_recall, average_precision, average_f1_score
