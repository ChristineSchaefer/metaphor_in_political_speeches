import os
from datetime import datetime

import evaluate
import numpy as np
import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field
from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers.utils import ModelOutput

from src.config import Settings, BASE_DIR
from src.data_handler.models.trofi_dataset import TroFiDataset
from src.mwe_metaphor.models.dataset_model import Dataset
from src.mwe_metaphor.models.evaluation_model import PredictionEvaluationModel
from src.mwe_metaphor.models.spacy_model import SpacyModel
from src.mwe_metaphor.utils.tsvlib import TSVSentence, iter_tsv_sentences


class PredictionController(BaseModel):
    settings: Settings
    test_dataset_mwe: Dataset = Field(default_factory=Dataset)
    test_dataset_metaphor: Dataset = Field(default_factory=Dataset)
    pre_training: bool
    num_epochs: int = 1
    evaluation_results: list[PredictionEvaluationModel] = Field(default_factory=list)

    def predict(self):
        self.preprocessing()
        model = AutoModelForTokenClassification.from_pretrained(self._get_latest_save(
            self.settings.model)) if self.pre_training else AutoModelForTokenClassification.from_pretrained(
            self.settings.model)
        tokenizer = AutoTokenizer.from_pretrained(
            self._get_latest_save(self.settings.model)) if self.pre_training else AutoTokenizer.from_pretrained(
            self.settings.model)

        self._evaluate_and_print("metaphor", self.test_dataset_metaphor, tokenizer, model)
        self._evaluate_and_print("mwe", self.test_dataset_mwe, tokenizer, model)

    def _evaluate_and_print(self, corpus_name, test_dataset, tokenizer, model):
        print(f"start evaluation for {corpus_name} corpus")
        for _ in range(self.num_epochs):
            evaluation_results = self.evaluate(test_dataset, tokenizer, model)
            self.evaluation_results.append(evaluation_results)
            print(f"epoch {_}: {evaluation_results}")

        averages = self._compute_average()
        print(f"{corpus_name} average over {self.num_epochs} epochs: {averages}")
        
    def _load_tsv_data(self) -> list[TSVSentence]:
        with open(f"{self.settings.mwe_dir}/{self.settings.mwe_test}") as f:
            return list(iter_tsv_sentences(f))

    def preprocessing(self):
        language_model = SpacyModel(language_model=self.settings.language_model).get_language_model()
        test_data_tsv_sentences = self._load_tsv_data()
        self.test_dataset_mwe.create_from_tsv(test_data_tsv_sentences)
        self.test_dataset_metaphor.create_from_trofi(TroFiDataset.find(), language_model)

    def _get_latest_save(self, model_name: str) -> str | None:
        subfolders = [f for f in os.listdir(os.path.join(BASE_DIR, self.settings.model_dir)) if
                      os.path.isdir(os.path.join(BASE_DIR, self.settings.model_dir, f))]

        if not subfolders:
            print("No subfolders found.")
            return None

        newest_subfolder = None
        newest_timestamp = datetime.min

        for subfolder in subfolders:
            try:
                model, timestamp_str = subfolder.split('_')
                if model != model_name:
                    continue
                timestamp = datetime.strptime(timestamp_str, "%Y%m%d%H%M%S")
                if timestamp > newest_timestamp:
                    newest_timestamp = timestamp
                    newest_subfolder = subfolder
            except ValueError:
                print(f"Ignoring invalid subfolder format: {subfolder}")

        return os.path.join(BASE_DIR, self.settings.model_dir, newest_subfolder)

    def evaluate(self, dataset: Dataset, tokenizer, model):
        token = next((column for column in dataset.columns if column.name == "tokens"), None)
        labels = next((column for column in dataset.columns if column.name == "label"), None)

        inputs = tokenizer(
            token.data,
            return_tensors="pt",
            padding=True,
            truncation=True,
            is_split_into_words=True)
        new_labels = []
        for i, label in enumerate(labels.data):
            word_ids = inputs.word_ids(i)
            new_labels.append(self.test_dataset_metaphor.align_labels_with_tokens(label, word_ids))
        inputs["labels"] = torch.tensor(new_labels)

        with torch.no_grad():
            outputs = model(**inputs)

        return self._compute_metrics(outputs, inputs["labels"])

    def _compute_metrics(self, eval_preds: ModelOutput, labels):
        metric = evaluate.load("seqeval")
        logits = eval_preds.logits
        probabilities = F.softmax(logits, dim=-1)
        predicted_labels = torch.argmax(probabilities, dim=-1)

        # Remove ignored index (special tokens) and convert to labels
        true_predictions = [
            [self.test_dataset_metaphor.labels[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predicted_labels, labels)
        ]
        true_labels = [
            [self.test_dataset_metaphor.labels[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predicted_labels, labels)
        ]

        all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
        return PredictionEvaluationModel(
            precision=all_metrics["overall_precision"],
            recall=all_metrics["overall_recall"],
            f1_score=all_metrics["overall_f1"],
            accuracy=all_metrics["overall_accuracy"])

    def _compute_average(self):
        precision_scores = [value.precision for value in self.evaluation_results]
        average_precision = np.mean(precision_scores)

        recall_scores = [value.recall for value in self.evaluation_results]
        average_recall = np.mean(recall_scores)

        f1_scores = [value.f1_score for value in self.evaluation_results]
        average_f1_score = np.mean(f1_scores)

        accuracy_score = [value.accuracy for value in self.evaluation_results]
        average_accuracy = np.mean(accuracy_score)

        return average_accuracy, average_recall, average_precision, average_f1_score
