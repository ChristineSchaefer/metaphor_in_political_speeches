import scipy
import sklearn_crfsuite
from pydantic import BaseModel, Field
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV
from sklearn_crfsuite import metrics

from src.config import Settings
from src.data_handler.models.trofi_dataset import TroFiDataset
from src.mwe_metaphor.utils.tsvlib import TSVSentence, iter_tsv_sentences
from src.utils.text_handler import normalize


class CRFController(BaseModel):
    settings: Settings
    train_data_sentences: list[list] = Field(default_factory=list)
    val_data_sentences: list[list] = Field(default_factory=list)
    test_mwe_data_sentences: list[list] = Field(default_factory=list)
    test_metaphor_data_sentences: list[list] = Field(default_factory=list)

    def build_crf(self):
        self.preprocessing()
        X_train = [sent2features(s) for s in self.train_data_sentences]
        y_train = [sent2labels(s) for s in self.train_data_sentences]

        # X_val = [sent2features(s) for s in self.val_data_sentences]
        # y_val = [sent2labels(s) for s in self.val_data_sentences]

        X_test_mwe = [sent2features(s) for s in self.test_mwe_data_sentences]
        y_test_mwe = [sent2labels(s) for s in self.test_mwe_data_sentences]

        X_test_metaphor = [sent2features(s) for s in self.test_metaphor_data_sentences]
        y_test_metaphor = [sent2labels(s) for s in self.test_metaphor_data_sentences]

        labels = ["is_metaphor", "no_metaphor"]

        # define fixed parameters and parameters to search
        crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            max_iterations=100,
            all_possible_transitions=True
        )
        params_space = {
            'c1': scipy.stats.expon(scale=0.5),
            'c2': scipy.stats.expon(scale=0.05),
        }

        # use the same metric for evaluation
        f1_scorer = make_scorer(metrics.flat_f1_score,
                                average='weighted', labels=labels)

        # search
        rs = RandomizedSearchCV(crf, params_space,
                                cv=3,
                                verbose=1,
                                n_jobs=-1,
                                n_iter=50,
                                scoring=f1_scorer,
                                return_train_score=True)
        rs.fit(X_train, y_train)

        print('best params:', rs.best_params_)
        print('best CV score:', rs.best_score_)
        print('model size: {:0.2f}M'.format(rs.best_estimator_.size_ / 1000000))

        crf = rs.best_estimator_

        y_pred = crf.predict(X_test_mwe)
        f1_score = metrics.flat_f1_score(y_test_mwe, y_pred,
                                         average='weighted', labels=labels)
        print(f"predicted mwe f1-score: {f1_score}")

        precision = metrics.flat_precision_score(y_test_mwe, y_pred, labels=labels, pos_label="is_metaphor")
        print(f"predicted mwe precision: {precision}")

        recall = metrics.flat_recall_score(y_test_mwe, y_pred, labels=labels, pos_label="is_metaphor")
        print(f"predicted mwe recall: {recall}")

        accuracy = metrics.flat_accuracy_score(y_test_mwe, y_pred)
        print(f"predicted mwe accuracy: {accuracy}")

        y_pred = crf.predict(X_test_metaphor)
        f1_score = metrics.flat_f1_score(y_test_metaphor, y_pred,
                                         average='weighted', labels=labels)
        print(f"predicted metaphor f1-score: {f1_score}")

        precision = metrics.flat_precision_score(y_test_metaphor, y_pred, labels=labels, pos_label="is_metaphor")
        print(f"predicted metaphor precision: {precision}")

        recall = metrics.flat_recall_score(y_test_metaphor, y_pred, labels=labels, pos_label="is_metaphor")
        print(f"predicted metaphor recall: {recall}")

        accuracy = metrics.flat_accuracy_score(y_test_metaphor, y_pred)
        print(f"predicted metaphor accuracy: {accuracy}")

    def _load_data(self, path: str) -> list[TSVSentence]:
        with open(f"{self.settings.mwe_dir}/{path}") as f:
            return list(iter_tsv_sentences(f))

    def preprocessing(self):
        tsv_train = self._load_data(self.settings.mwe_train)
        tsv_test = self._load_data(self.settings.mwe_test)
        tsv_val = self._load_data(self.settings.mwe_val)

        self.train_data_sentences = self._process_tsv_data(tsv_train)
        self.test_mwe_data_sentences = self._process_tsv_data(tsv_test)
        self.val_data_sentences = self._process_tsv_data(tsv_val)
        self.test_metaphor_data_sentences = self._process_metaphor_data(TroFiDataset.find())

    @staticmethod
    def _process_tsv_data(data):
        processed_data = []
        for sentence in data:
            sentence_list = [
                (words.get("FORM", "0"),
                 "no_metaphor" if words.get("PARSEME:MWE", "no_metaphor") == "*" else "is_metaphor") for
                words in sentence.words]
            processed_data.append(sentence_list)
        return processed_data

    @staticmethod
    def _process_metaphor_data(data: list[TroFiDataset]):
        processed_data = []
        for sentence in data:
            token = normalize(sentence.sentence).split()
            sentence_list = [(t, "no_metaphor") if index != sentence.verb_idx else (t, "is_metaphor") for index, t in
                             enumerate(token)]
            processed_data.append(sentence_list)
        return processed_data


def word2features(sent, i):
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


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [label for token, label in sent]
