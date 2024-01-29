import os
from datetime import datetime

import evaluate
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
from src.utils.datetime import ts_now
from src.utils.text_handler import write_list_with_dict_to_txt


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
        return self.evaluation_results

    def _evaluate_and_print(self, corpus_name, test_dataset, tokenizer, model):
        print(f"start evaluation for {corpus_name} corpus")
        evaluation_results = self.evaluate(test_dataset, tokenizer, model)
        self.evaluation_results.append(evaluation_results)
        print(f"epoch {self.num_epochs}: {evaluation_results}")
        
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

        return self._compute_metrics(outputs, inputs, dataset)

    @classmethod
    def _compute_metrics(cls, eval_preds: ModelOutput, inputs, dataset: Dataset):
        metric = evaluate.load("seqeval")
        logits = eval_preds.logits
        probabilities = F.softmax(logits, dim=-1)
        predicted_labels = torch.argmax(probabilities, dim=-1)

        # Remove ignored index (special tokens) and convert to labels
        true_predictions = [
            [dataset.labels[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predicted_labels, inputs["labels"])
        ]
        true_labels = [
            [dataset.labels[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predicted_labels, inputs["labels"])
        ]

        wrong_predictions = cls._get_sentence_with_wrong_prediction(true_labels, true_predictions, dataset, inputs.word_ids)
        write_list_with_dict_to_txt(wrong_predictions, f"data/logs/{ts_now()}_wrong_predictions.txt", "w")
        all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
        return PredictionEvaluationModel(
            precision=all_metrics["overall_precision"],
            recall=all_metrics["overall_recall"],
            f1_score=all_metrics["overall_f1"],
            accuracy=all_metrics["overall_accuracy"])

    @classmethod
    def _get_sentence_with_wrong_prediction(
            cls,
            true_labels: list,
            predictions: list,
            dataset: Dataset,
            word_ids) -> list:
        wrong_predictions = []
        token_column = [t.data for t in dataset.columns if t.name == "tokens"]
        for index, pred in enumerate(predictions):
            if pred != true_labels[index]:
                wrong_predictions.append({
                    "predictions": cls._normalize_labels(pred, word_ids(index)),
                    "true_labels": cls._normalize_labels(true_labels[index], word_ids(index)),
                    "token": token_column[0][index]
                })
        return wrong_predictions

    @classmethod
    def _normalize_labels(cls, labels: list, word_ids: list) -> list:
        normalized_labels = []

        current_word = None
        for id in (i for i in word_ids if i is not None):
            if current_word is None or current_word != id:
                normalized_labels.append(labels.pop(0))
            current_word = id

        return normalized_labels


