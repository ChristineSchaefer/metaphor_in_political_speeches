import os
import re

import evaluate
import numpy as np
import pandas as pd
import torch
from evaluate import evaluator
from pydantic import BaseModel, Field
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer

from src.config import Settings, get_settings, BASE_DIR
from src.mwe_metaphor.models.dataset_model import Dataset, MWEDataset
from src.mwe_metaphor.utils.tsvlib import iter_tsv_sentences, TSVSentence

device = torch.device("mps") if torch.has_mps else torch.device("cpu")


class BertTrainingController(BaseModel):
    settings: Settings
    train_data_sentences: list[TSVSentence] = Field(default_factory=list)
    test_data_sentences: list[TSVSentence] = Field(default_factory=list)
    val_data_sentences: list[TSVSentence] = Field(default_factory=list)
    train_dataset: Dataset = Field(default_factory=Dataset)
    test_dataset: Dataset = Field(default_factory=Dataset)
    val_dataset: Dataset = Field(default_factory=Dataset)
    labels: list[str] = Field(default_factory=list)

    def training(self):
        # load datasets and preprocessing
        self.preprocessing()

        # processing the data
        model_checkpoint = "distilbert-base-german-cased"
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        # tokenize input and create word_ids, extend labels with special tokens
        tokenized_inputs_train = self.tokenize_and_align_labels(self.train_dataset, tokenizer)
        tokenized_inputs_test = self.tokenize_and_align_labels(self.test_dataset, tokenizer)
        tokenized_inputs_val = self.tokenize_and_align_labels(self.val_dataset, tokenizer)

        train_dataset = MWEDataset(tokenized_inputs_train, tokenized_inputs_train.data["labels"])
        test_dataset = MWEDataset(tokenized_inputs_test, tokenized_inputs_test.data["labels"])
        val_dataset = MWEDataset(tokenized_inputs_val, tokenized_inputs_val.data["labels"])

        model = AutoModelForTokenClassification.from_pretrained(
            model_checkpoint,
            num_labels=len(self.train_dataset.labels),
            ignore_mismatched_sizes=True)

        model.save_pretrained(os.path.join(BASE_DIR, f"data/models/{model_checkpoint}"))
        tokenizer.save_pretrained(os.path.join(BASE_DIR, f"data/models/{model_checkpoint}"))

        training_args = TrainingArguments(
            output_dir='./results',  # output directory
            num_train_epochs=3,  # total number of training epochs
            per_device_train_batch_size=16,  # batch size per device during training
            per_device_eval_batch_size=64,  # batch size for evaluation
            warmup_steps=500,  # number of warmup steps for learning rate scheduler
            weight_decay=0.01,  # strength of weight decay
            logging_dir='./logs',  # directory for storing logs
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5
        )

        trainer = Trainer(
            model=model,  # the instantiated 🤗 Transformers model to be trained
            args=training_args,  # training arguments, defined above
            train_dataset=train_dataset,  # training dataset
            eval_dataset=val_dataset,  # evaluation dataset
            compute_metrics=self.compute_metrics
        )

        trainer.train()

    def load_data(self, path: str) -> list[TSVSentence]:
        with open(f"{self.settings.mwe_dir}/{path}") as f:
            return list(iter_tsv_sentences(f))

    def preprocessing(self):
        self.train_data_sentences = self.load_data(self.settings.mwe_train)
        self.test_data_sentences = self.load_data(self.settings.mwe_test)
        self.val_data_sentences = self.load_data(self.settings.mwe_val)

        self.train_dataset.create(self.train_data_sentences)
        self.test_dataset.create(self.test_data_sentences)
        self.val_dataset.create(self.val_data_sentences)

    @staticmethod
    def align_labels_with_tokens(labels, word_ids):
        new_labels = []
        current_word = None
        for word_id in word_ids:
            if word_id != current_word:
                # start of a new word
                current_word = word_id
                label = "-100" if word_id is None else labels[word_id]
                new_labels.append(label)
            elif word_id is None:
                # special token
                new_labels.append("-100")
            else:
                # same word as previous token
                label = labels[word_id]
                # only the identifier is used for a continuous word
                pattern = re.compile(r'^(\d+):.*$')
                match = pattern.match(label)
                if match:
                    label = match.group(1)
                new_labels.append(label)

        return new_labels

    def tokenize_and_align_labels(self, dataset: Dataset, tokenizer):
        tokenized_inputs = tokenizer(
            dataset.columns[1].data,
            padding="max_length",
            truncation=True,
            is_split_into_words=True
        )
        all_labels = dataset.columns[-1].data
        new_labels = []
        for i, labels in enumerate(all_labels):
            word_ids = tokenized_inputs.word_ids(i)
            new_labels.append(self.align_labels_with_tokens(labels, word_ids))

        new_labels = dataset.refactor_labels_columns(new_labels)
        tokenized_inputs["labels"] = new_labels
        return tokenized_inputs

    def compute_metrics(self, eval_preds):
        # TODO check evaluation function and labels because no ner task
        metric = evaluate.load("seqeval")
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)

        # Remove ignored index (special tokens) and convert to labels
        true_labels = [[self.train_dataset.id2label[l] for l in label] for label in labels]
        true_predictions = [[self.train_dataset.id2label[p] for p in pred] for pred in predictions]
        all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": all_metrics["overall_precision"],
            "recall": all_metrics["overall_recall"],
            "f1": all_metrics["overall_f1"],
            "accuracy": all_metrics["overall_accuracy"],
        }

    @staticmethod
    def postprocess(predictions, labels, label_names):
        predictions = predictions.detach().cpu().clone().numpy()
        labels = labels.detach().cpu().clone().numpy()

        # Remove ignored index (special tokens) and convert to labels
        true_labels = [[label_names[l] for l in label if l != "-100"] for label in labels]
        true_predictions = [
            [label_names[p] for (p, l) in zip(prediction, label) if l != "-100"]
            for prediction, label in zip(predictions, labels)
        ]
        return true_labels, true_predictions


if __name__ == '__main__':
    controller = BertTrainingController(settings=get_settings())
    controller.training()

