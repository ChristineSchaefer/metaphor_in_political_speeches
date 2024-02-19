from itertools import chain

import pycrfsuite
import scipy
from pydantic import BaseModel, Field
from sklearn.metrics import make_scorer, classification_report
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelBinarizer

from src.config import Settings
from src.data_handler.models.trofi_dataset import TroFiDataset
from src.mwe_metaphor.utils.text_utils import load_data
from src.utils.text_handler import normalize


class CRFController(BaseModel):
    """
        The CRFController class handles the configuration and operation of a Conditional Random Field (CRF) model.
    """
    settings: Settings = Field(..., description="project settings")
    train_data_sentences: list[list] = Field(default_factory=list, description="train data sentences")
    val_data_sentences: list[list] = Field(default_factory=list, description="val data sentences")
    test_mwe_data_sentences: list[list] = Field(default_factory=list, description="test data sentences for mwe")
    test_metaphor_data_sentences: list[list] = Field(default_factory=list, description="test data sentences for metaphor")

    def build_crf(self):
        """
            Constructs and configures the CRF model. Main method to run the CRF.
        """

        # prepare and create data
        self.preprocessing()
        X_train = [sent2features(s) for s in self.train_data_sentences]
        y_train = [sent2labels(s) for s in self.train_data_sentences]

        # not necessary for used scenario
        # X_val = [sent2features(s) for s in self.val_data_sentences]
        # y_val = [sent2labels(s) for s in self.val_data_sentences]

        X_test_mwe = [sent2features(s) for s in self.test_mwe_data_sentences]
        y_test_mwe = [sent2labels(s) for s in self.test_mwe_data_sentences]

        X_test_metaphor = [sent2features(s) for s in self.test_metaphor_data_sentences]
        y_test_metaphor = [sent2labels(s) for s in self.test_metaphor_data_sentences]

        # label for binary classification
        labels = ["is_metaphor", "no_metaphor"]

        trainer = pycrfsuite.Trainer(verbose=False)

        for xseq, yseq in zip(X_train, y_train):
            trainer.append(xseq, yseq)

        trainer.set_params({
            'c1': scipy.stats.expon(scale=0.5),  # coefficient for L1 penalty
            'c2': scipy.stats.expon(scale=0.05),  # coefficient for L2 penalty
            'max_iterations': 100,  # stop earlier

            # include transitions that are possible, but not observed
            'feature.possible_transitions': True
        })

        trainer.train('mwe_metaphor.crfsuite')

        tagger = pycrfsuite.Tagger()
        tagger.open('mwe_metaphor.crfsuite')

        # mwe testing
        y_pred_mwe = [tagger.tag(xseq) for xseq in X_test_mwe]
        print(f"CRF prediction evaluation results for mwe: \n {self.bio_classification_report(y_test_mwe, y_pred_mwe)}")

        # metaphor testing
        y_pred_metaphor = [tagger.tag(xseq) for xseq in X_test_metaphor]
        print(f"CRF prediction evaluation results for metaphor: \n {self.bio_classification_report(y_test_metaphor, y_pred_metaphor)}")

    @staticmethod
    def bio_classification_report(y_true, y_pred):
        """
            Classification report for a list of BIO-encoded sequences.
            It computes token-level metrics and discards "O" labels.

            @param y_true: true labels
            @param y_pred: predicted labels

            @returns classification report
        """
        lb = LabelBinarizer()
        y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
        y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))

        tagset = set(lb.classes_) - {'O'}
        tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
        class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}

        return classification_report(
            y_true_combined,
            y_pred_combined,
            labels=[class_indices[cls] for cls in tagset],
            target_names=tagset,
        )

    def preprocessing(self):
        """
           Responsible for preprocessing the loaded data.
           Converts the data into a format that is suitable for CRF training.
        """
        tsv_train = load_data(self.settings, self.settings.mwe_train)
        tsv_test = load_data(self.settings, self.settings.mwe_test)
        tsv_val = load_data(self.settings, self.settings.mwe_val)

        self.train_data_sentences = self._process_tsv_data(tsv_train)
        self.test_mwe_data_sentences = self._process_tsv_data(tsv_test)
        self.val_data_sentences = self._process_tsv_data(tsv_val)
        self.test_metaphor_data_sentences = self._process_metaphor_data(TroFiDataset.find())

    @staticmethod
    def _process_tsv_data(data: list) -> list:
        """
           Responsible for preprocessing the tsv data.

           @param data: list with tsv sentences
           @return list tupel
        """
        processed_data = []
        for sentence in data:
            sentence_list = [
                (words.get("FORM", "0"),
                 "no_metaphor" if words.get("PARSEME:MWE", "no_metaphor") == "*" else "is_metaphor") for
                words in sentence.words]
            processed_data.append(sentence_list)
        return processed_data

    @staticmethod
    def _process_metaphor_data(data: list[TroFiDataset]) -> list:
        """
           Responsible for preprocessing the metaphor data.

           @param data: list with trofi data
           @return list with tupel
        """
        processed_data = []
        for sentence in data:
            token = normalize(sentence.sentence).split()
            sentence_list = [(t, "no_metaphor") if index != sentence.verb_idx else (t, "is_metaphor") for index, t in
                             enumerate(token)]
            processed_data.append(sentence_list)
        return processed_data


def word2features(sent: list, i: int) -> dict:
    """
        Create word to features vector for sentences in data.

        @param sent: list with sentences
        @param i: index

        @return dictionary with crf features
    """
    word = sent[i][0]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit()
    }
    if i > 0:
        word1 = sent[i - 1][0]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper()
        })
    else:
        features['BOS'] = True

    if i < len(sent) - 1:
        word1 = sent[i + 1][0]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent: list) -> list:
    """
        Create list with sentences and features.

        @param sent: list of sentences
        @return list with dict for sentences
    """
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent: list) -> list:
    """
        Create list with label per sentence.

        @param sent: list of sentences
        @return list with labels
    """
    return [label for token, label in sent]
