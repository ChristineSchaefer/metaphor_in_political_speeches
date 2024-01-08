import torch
from pydantic import BaseModel, Field

from src.mwe_metaphor.utils.tsvlib import TSVSentence


class Column(BaseModel):
    name: str
    data: list


class Feature(BaseModel):
    feature: str
    names: list[str] = Field(default_factory=list)


class Dataset(BaseModel):
    features: list[Feature] | None = Field(default_factory=list)
    num_rows: int = 0
    columns: list[Column] | None = Field(default_factory=list)
    id2label: dict | None = Field(default_factory=dict)
    label2id: dict | None = Field(default_factory=dict)
    labels: list | None = Field(default_factory=list)

    def add_column(self, column: Column):
        self.columns.append(column)

    def add_columns(self, columns: list[Column]):
        for column in columns:
            self.add_column(column)

    def remove_column(self, column_name: str):
        for column in self.columns:
            if column.name == column_name:
                self.columns.remove(column)
            else:
                continue

    def set_rows(self):
        self.num_rows = len(self.columns[0].data)

    def set_features(self):
        for column in self.columns:
            available_feature = set()
            for data in column.data:
                if isinstance(data, list):
                    available_feature.update(d for d in data)
                else:
                    available_feature.add(data)
            self.features.append(Feature(feature=column.name, names=list(available_feature)))

    def get_column_data(self, column: int):
        return self.columns[column].data

    def create(self, sentences: list[TSVSentence]):
        id_array, token_array, lemma_array, upos_array, xpos_array = [], [], [], [], []
        deprel_array, head_array, parseme_mwe_array = [], [], []

        for id, sentence in enumerate(sentences):
            tokens = [words.get("FORM", "0") for words in sentence.words]
            lemmas = [words.get("LEMMA", "0") for words in sentence.words]
            upos = [words.get("UPOS", "0") for words in sentence.words]
            xpos = [words.get("XPOS", "0") for words in sentence.words]
            heads = [words.get("HEAD", "0") for words in sentence.words]
            deprels = [words.get("DEPREL", "0") for words in sentence.words]
            parseme_mwes = ["no_metaphor" if words.get("PARSEME:MWE", "0") == "*" else "is_metaphor" for words in sentence.words]

            id_array.append(id)
            token_array.append(tokens)
            lemma_array.append(lemmas)
            upos_array.append(upos)
            xpos_array.append(xpos)
            head_array.append(heads)
            deprel_array.append(deprels)
            parseme_mwe_array.append(parseme_mwes)

        columns = [
            Column(name="id", data=id_array),
            Column(name="tokens", data=token_array),
            Column(name="lemmas", data=lemma_array),
            Column(name="upos", data=upos_array),
            Column(name="xpos", data=xpos_array),
            Column(name="heads", data=head_array),
            Column(name="deprels", data=deprel_array),
            Column(name="parseme_mwes", data=parseme_mwe_array),
        ]

        self.add_columns(columns)
        self.set_rows()
        self.set_features()

    def refactor_labels_columns(self, labels: list[list[str]]) -> list[list[int]]:
        index_to_word = self.create_word_to_index_voc(labels)
        return [
            [list(index_to_word.keys())[list(index_to_word.values()).index(word)] for word in label]
            for label in labels
        ]

    def create_word_to_index_voc(self, labels: list[list[str]]):
        voc = set(word for label in labels for word in label)
        word_to_index = {index: word for index, word in enumerate(list(voc), start=1) if word != "-100"}
        word_to_index[0] = "-100"
        self.id2label = word_to_index
        self.label2id = {id: tag for tag, id in self.id2label.items()}
        self.labels = list(word_to_index.values())
        return word_to_index


class MWEDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

