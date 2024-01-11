import os
from datetime import datetime

import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field
from transformers import AutoModelForTokenClassification, AutoTokenizer

from src.config import Settings, get_settings, BASE_DIR
from src.data_handler.models.trofi_dataset import TroFiDataset
from src.mwe_metaphor.models.dataset_model import Dataset
from src.mwe_metaphor.models.spacy_model import SpacyModel
from src.mwe_metaphor.utils.tsvlib import TSVSentence, iter_tsv_sentences


class PredictionController(BaseModel):
    settings: Settings
    test_dataset_mwe: Dataset = Field(default_factory=Dataset)
    test_dataset_metaphor: Dataset = Field(default_factory=Dataset)

    def predict(self):
        self.preprocessing()
        model = AutoModelForTokenClassification.from_pretrained(self.get_latest_save(self.settings.model))
        tokenizer = AutoTokenizer.from_pretrained(self.get_latest_save(self.settings.model))

        inputs = tokenizer(self.test_dataset_metaphor.columns[0].data, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)

        predictions = outputs.logits

        probabilities = F.softmax(predictions, dim=-1)
        predicted_labels = torch.argmax(probabilities, dim=-1)

        # TODO predictions umwandeln in Klassen
        #   print anpassen

        for sentence, label in zip(self.test_dataset_metaphor.columns[0].data, predicted_labels.tolist()):
            print(f"Sentence: {sentence}, Predicted Label: {label}")

    def load_tsv_data(self, path: str) -> list[TSVSentence]:
        with open(f"{self.settings.mwe_dir}/{path}") as f:
            return list(iter_tsv_sentences(f))

    def preprocessing(self):
        language_model = SpacyModel(language_model=self.settings.language_model).get_language_model()
        test_data_tsv_sentences = self.load_tsv_data(self.settings.mwe_test)
        self.test_dataset_mwe.create_from_tsv(test_data_tsv_sentences)
        self.test_dataset_metaphor.create_from_trofi(TroFiDataset.find(), language_model)

    def get_latest_save(self, model_name: str) -> str | None:
        subfolders = [f for f in os.listdir(os.path.join(BASE_DIR, self.settings.model_dir)) if os.path.isdir(os.path.join(BASE_DIR, self.settings.model_dir, f))]

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


if __name__ == "__main__":
    pc = PredictionController(settings=get_settings())
    pc.predict()


