import os

import evaluate
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, \
    DataCollatorForTokenClassification

from src.config import Settings, BASE_DIR
from src.mwe_metaphor.models.dataset_model import Dataset, MWEDataset
from src.mwe_metaphor.utils.tsvlib import iter_tsv_sentences, TSVSentence
from src.mwe_metaphor.utils.visualisation import plot_history, process_and_chart
from src.utils.datetime import ts_now


class BertTrainingController(BaseModel):
    """
        This controller is responsible for training a BERT model.
        It handles data loading, preprocessing, training, and evaluation stages.
    """
    settings: Settings = Field(description="The configuration settings for training the model.")
    train_data_sentences: list[TSVSentence] = Field(default_factory=list, description="The training data used for model training. These are the sentences or phrases the model learns from.")
    val_data_sentences: list[TSVSentence] = Field(default_factory=list, description="The validation data used for validating the model's performance. These sentences or phrases are used to tune the model.")
    train_dataset: Dataset = Field(default_factory=Dataset, description="The actual training dataset, which might be different from train_data_sentences after preprocessing or other transformations.")
    val_dataset: Dataset = Field(default_factory=Dataset, description="The actual validation dataset, which might differ from val_data_sentences after preprocessing or other transformations.")
    labels: list[str] = Field(default_factory=list, description="The target labels/classes for the training data.")

    def training(self):
        """
            This method handles the training process of the BERT model using the settings and data-set defined in the instance.
            Includes configuration setup, model definition, training and validation.
        """
        # load datasets and preprocessing
        self.preprocessing()

        # processing the data
        model_checkpoint = self.settings.model
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        # tokenize input and create word_ids, extend labels with special tokens
        tokenized_inputs_train = self.train_dataset.tokenize_and_align_labels(tokenizer)
        tokenized_inputs_val = self.val_dataset.tokenize_and_align_labels(tokenizer)

        # create bar charts for datasets
        # process_and_chart(dataset=self.train_dataset, tokenized_inputs=tokenized_inputs_train, dataset_name="train_data")
        # process_and_chart(dataset=self.val_dataset, tokenized_inputs=tokenized_inputs_val, dataset_name="val_data")

        train_dataset = MWEDataset(tokenized_inputs_train, tokenized_inputs_train.data["labels"])
        val_dataset = MWEDataset(tokenized_inputs_val, tokenized_inputs_val.data["labels"])

        model = AutoModelForTokenClassification.from_pretrained(
            model_checkpoint,
            num_labels=len(self.train_dataset.labels),
            ignore_mismatched_sizes=True,
            id2label=self.train_dataset.id2label,
            label2id=self.train_dataset.label2id)

        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

        training_args = TrainingArguments(
            output_dir=f'./results/{model_checkpoint}_{ts_now()}',  # output directory
            num_train_epochs=self.settings.epochs,  # total number of training epochs
            per_device_train_batch_size=self.settings.batch_train,  # batch size per device during training
            per_device_eval_batch_size=64,  # batch size for evaluation
            warmup_steps=self.settings.num_warmup_steps,  # number of warmup steps for learning rate scheduler
            weight_decay=0.01,  # strength of weight decay
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            load_best_model_at_end=True
        )

        trainer = Trainer(
            model=model,  # the instantiated 🤗 Transformers model to be trained
            args=training_args,  # training arguments, defined above
            train_dataset=train_dataset,  # training dataset
            eval_dataset=val_dataset,  # evaluation dataset
            compute_metrics=self.compute_metrics,
            data_collator=data_collator,
            tokenizer=tokenizer
        )

        trainer.train()

        timestamp = ts_now()
        model.save_pretrained(os.path.join(BASE_DIR, self.settings.model_dir + f"{model_checkpoint}_{timestamp}"))
        tokenizer.save_pretrained(os.path.join(BASE_DIR, self.settings.model_dir + f"{model_checkpoint}_{timestamp}"))

        history = pd.DataFrame(trainer.state.log_history)
        history.to_csv(path_or_buf=os.path.join(BASE_DIR, f"data/logs/training_history/{ts_now()}_training_history.csv"))
        plot_history(history, model_checkpoint)
        print(history)

    def _load_data(self, path: str) -> list[TSVSentence]:
        """
            A private method that loads data for training and validation.
            It sets the data as attributes to the instance for later usage.

            @param path: path to the mwe korpus

            @returns list with TSVSentence objects
        """
        with open(f"{self.settings.mwe_dir}/{path}") as f:
            return list(iter_tsv_sentences(f))

    def preprocessing(self):
        """
           Responsible for preprocessing the loaded data.
           Converts the data into a format that is suitable for BERT training.
        """
        self.train_data_sentences = self._load_data(self.settings.mwe_train)
        self.val_data_sentences = self._load_data(self.settings.mwe_val)

        self.train_dataset.create_from_tsv(self.train_data_sentences)
        self.val_dataset.create_from_tsv(self.val_data_sentences)

    def compute_metrics(self, eval_preds):
        """
           Computes the metrics such as accuracy, precision, recall, etc.
           Useful to evaluate the performance of the model.

           @param eval_preds: the predictions from training

           @returns evaluation metrics dict
        """
        metric = evaluate.load("seqeval")
        predictions, labels = eval_preds
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens) and convert to labels
        true_predictions = [
            [self.train_dataset.labels[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.train_dataset.labels[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": all_metrics["overall_precision"],
            "recall": all_metrics["overall_recall"],
            "f1": all_metrics["overall_f1"],
            "accuracy": all_metrics["overall_accuracy"],
        }

    @staticmethod
    def postprocess(predictions, labels, label_names):
        """
            Responsible for any post-processing steps after training.
            Includes tasks like saving model parameters and making the model ready for predictions.

            @param labels: list of predicted labels
            @param label_names: list of label names

            @returns list with true labels and predictions
        """
        predictions = predictions.detach().cpu().clone().numpy()
        labels = labels.detach().cpu().clone().numpy()

        # Remove ignored index (special tokens) and convert to labels
        true_labels = [[label_names[l] for l in label if l != "-100"] for label in labels]
        true_predictions = [
            [label_names[p] for (p, l) in zip(prediction, label) if l != "-100"]
            for prediction, label in zip(predictions, labels)
        ]
        return true_labels, true_predictions
