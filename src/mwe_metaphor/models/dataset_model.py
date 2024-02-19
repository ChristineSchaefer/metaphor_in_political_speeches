import torch
from pydantic import BaseModel, Field

from src.config import Modus, get_settings
from src.data_handler.models.trofi_dataset import TroFiDataset
from src.mwe_metaphor.utils.tsvlib import TSVSentence
from src.utils.text_handler import normalize


class Column(BaseModel):
    """
        This class represents a column in a data set.
    """
    name: str = Field(description="The name of the column.")
    data: list = Field(description="The data contained in the column.")


class Feature(BaseModel):
    """
        This class represents a feature in a data set.
    """
    feature: str = Field(...,description="The specific feature type.")
    names: list[str] = Field(default_factory=list, description="The names of the individual features.")


class Dataset(BaseModel):
    """
        This class represents a dataset for training and evaluating ML models.
    """
    features: list[Feature] | None = Field(default_factory=list, description="The features included in the dataset.")
    num_rows: int = Field(default=0, description="The number of rows in the dataset.")
    columns: list[Column] | None = Field(default_factory=list, description="The columns in the dataset.")
    id2label: dict | None = Field(default_factory=dict, description=" A mapping from ID to label (for instance, for categorical variables).")
    label2id: dict | None = Field(default_factory=dict, description="A mapping from label to ID (inverse of id2label).")
    labels: list | None = Field(default_factory=list, description="The labels/classes for each instance/sample in the data.")
    modus: Modus = Field(default=get_settings().modus, description="The modus used for classification")

    def add_column(self, column: Column):
        """
            Add a new column to the dataset.

            @param column: An instance of the Column class.
        """
        self.columns.append(column)

    def add_columns(self, columns: list[Column]):
        """
            Add multiple new columns to the dataset.

            @param columns: A list of instances from the Column class.
        """
        for column in columns:
            self.add_column(column)

    def remove_column(self, column_name: str):
        """
            Remove a column from the dataset.

            @param column_name: The name of Column to be removed.
        """
        for column in self.columns:
            if column.name == column_name:
                self.columns.remove(column)
            else:
                continue

    def set_rows(self):
        """
            Set the number of rows (instances/samples) for the dataset.
        """
        self.num_rows = len(self.columns[0].data)

    def set_features(self):
        """
            Set the features for the dataset.
        """
        for column in self.columns:
            available_feature = set()
            for data in column.data:
                if isinstance(data, list):
                    available_feature.update(d for d in data)
                else:
                    available_feature.add(data)
            self.features.append(Feature(feature=column.name, names=list(map(str, available_feature))))

    def get_column_data(self, column: int):
        """
            Retrieve the data from a given column.

            @param column: The index of the column

            @returns list with column data
        """
        return self.columns[column].data

    def create_from_tsv(self, sentences: list[TSVSentence]):
        """
            Create a new Dataset object from TSVSentence.

            @param sentences: A list of TSVSentence
        """
        id_array, token_array, lemma_array, upos_array, xpos_array = [], [], [], [], []
        deprel_array, head_array, parseme_mwe_array = [], [], []

        for id, sentence in enumerate(sentences):
            tokens = [words.get("FORM", "0") for words in sentence.words]
            lemmas = [words.get("LEMMA", "0") for words in sentence.words]
            upos = [words.get("UPOS", "0") for words in sentence.words]
            xpos = [words.get("XPOS", "0") for words in sentence.words]
            heads = [words.get("HEAD", "0") for words in sentence.words]
            deprels = [words.get("DEPREL", "0") for words in sentence.words]
            # B-MET=Begin Metaphor, I-MET=Inner-Metaphor, 0=No Metaphor
            if self.modus.BINARY:
                parseme_mwes = [0 if words.get("PARSEME:MWE", 0) == "*" else 1 for words in sentence.words]
            else:
                parseme_mwes = [2 if words.get("PARSEME:MWE", 0).isdigit() else 1 if words.get("PARSEME:MWE", 0) != "*" else 0 for words in sentence.words]
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
            Column(name="label", data=parseme_mwe_array),
        ]

        self.add_columns(columns)
        self.set_rows()
        self.set_features()
        if self.modus == Modus.BINARY:
            self.labels = ["no_metaphor", "is_metaphor"]
        else:
            self.labels = ["O", "B-MET", "I-MET"]
        self.id2label = {index: label for index, label in enumerate(self.labels)}
        self.label2id = {id: tag for tag, id in self.id2label.items()}

    def create_from_trofi(self, sentences: list[TroFiDataset], language_model):
        """
            Create a new Dataset object from TroFi data.

            @param sentences: A list of TroFi data
            @param language_model: A Spacy language model for lemmatization
        """
        sentence_array, lemma_array, token_array, label_array = [], [], [], []
        for sentence in sentences:
            token = normalize(sentence.sentence).split()
            lemma = []
            for t in token:
                doc = language_model(t)
                result = ' '.join([x.lemma_ for x in doc])
                lemma.append(result)
            label = [0] * len(token)
            label[sentence.verb_idx] = 0 if sentence.label == 0 else 1

            sentence_array.append(sentence.sentence)
            lemma_array.append(lemma)
            label_array.append(label)
            token_array.append(token)

        columns = [
            Column(name="sentences", data=sentence_array),
            Column(name="label", data=label_array),
            Column(name="tokens", data=token_array),
            Column(name="lemmas", data=lemma_array),
        ]

        self.add_columns(columns)
        self.set_rows()
        self.set_features()
        if self.modus == Modus.BINARY:
            self.labels = ["no_metaphor", "is_metaphor"]
        else:
            self.labels = ["O", "B-MET", "I-MET"]
        self.id2label = {index: label for index, label in enumerate(self.labels)}
        self.label2id = {id: tag for tag, id in self.id2label.items()}

    def align_labels_with_tokens(self, labels, word_ids):
        """
            Align labels with tokens in the dataset.

            @param labels: list of old labels
            @param word_ids: list of word ids

            @returns list with segmented labels
        """
        new_labels = []
        current_word = None
        for word_id in word_ids:
            if word_id != current_word:
                # start of a new word
                current_word = word_id
                label = -100 if word_id is None else labels[word_id]
                new_labels.append(label)
            elif word_id is None:
                # special token
                new_labels.append(-100)
            else:
                # same word as previous token
                label = labels[word_id]
                # BIO: if label is beginning, then next label is inner (for BIO-tag)
                if self.modus == Modus.MULTI_LABEL:
                    if label == 1:
                        label = 2
                new_labels.append(label)

        return new_labels

    def tokenize_and_align_labels(self, tokenizer):
        """
            Tokenize the data and align labels with the tokens.

            @param tokenizer: AutoTokenizer Object

            @returns tokenized input
        """
        token = next((column for column in self.columns if column.name == "tokens"), None)
        labels = next((column for column in self.columns if column.name == "label"), None)
        try:
            tokenized_inputs = tokenizer(
                token.data,
                padding="max_length",
                truncation=True,
                is_split_into_words=True
            )
            new_labels = []
            for i, label in enumerate(labels.data):
                word_ids = tokenized_inputs.word_ids(i)
                new_labels.append(self.align_labels_with_tokens(label, word_ids))

            # new_labels = self.refactor_labels_columns(new_labels)
            tokenized_inputs["labels"] = new_labels
            return tokenized_inputs

        except ValueError as e:
            print(e)


class MWEDataset(torch.utils.data.Dataset):
    """
        This class represents a dataset specifically designed for Multi-Word Expressions (MWE).
    """
    def __init__(self, encodings, labels):
        """
            Initialize the MWEDataset object.
        """
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        """
            Retrieve an item from the dataset using its index.

            @param idx: index for label

            @returns item
        """
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        """
            Get the length (number of items/samples) of the dataset.

            @returns length of labels
        """
        return len(self.labels)
