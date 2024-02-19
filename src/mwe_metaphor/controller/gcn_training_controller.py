import gc

import torch
from pydantic import BaseModel, Field, computed_field
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from torch.optim import AdamW
from torch.utils.data import TensorDataset, RandomSampler, DataLoader, SequentialSampler
from tqdm import trange
from transformers import BertConfig, get_linear_schedule_with_warmup, AutoTokenizer

from src.config import Settings
from src.mwe_metaphor.models.bert_with_gcn_model import BertWithGCNAndMWE
from src.mwe_metaphor.models.evaluation_model import Evaluate
from src.mwe_metaphor.models.spacy_model import CoNLLParserModel, SpacyModel
from src.mwe_metaphor.utils.text_utils import tokenize
from src.mwe_metaphor.utils.training_utils import adjacency, pad_or_truncate, mwe_adjacency


class TrainingController(BaseModel):
    """
        The TrainingController class handles BERT with GCN training-related activities.

        from https://github.com/omidrohanian/metaphor_mwe/blob/master/main.py
    """
    settings: Settings = Field(..., description="project settings")
    max_grad_norm: float = Field(default=1.0, description="maximum gradient norm values")
    max_len: int = Field(default=0, description="maximum sequence length")
    all_test_indices: list = Field(default_factory=list, description="list with test indices")
    all_predictions: list = Field(default_factory=list, description="list with predictions")
    all_folds_labels: list = Field(default_factory=list, description="list with fold labels")
    recorded_results_per_fold: list = Field(default_factory=list, description="list with results per fold")
    random_seed: int = Field(default=616, description="random seed")

    @computed_field
    @property
    def device(self):
        return torch.device(self.settings.device)

    def training(self):
        """
            The top-level method that coordinates the training process.

            This method involves loading data, preparing the configuration, and triggering the training loop.

            @returns evaluation results
        """

        print(f"+++ start BERT with GCN training on {self.device} +++")

        # set adjacency matrix and other vectors
        A, A_MWE, labels, target_token_indices, vocab, texts = self.prepare_data()
        input_id, bert_config = self.prepare_bert_config(texts, vocab)

        # create training and test folds
        splits = self.train_test_loader(
            input_id,
            labels,
            A,
            A_MWE,
            target_token_indices,
            self.settings.K,
            self.settings.batch_train,
            self.settings.batch_test)

        # start training
        for i, (train_dataloader, test_dataloader) in enumerate(splits):
            model = BertWithGCNAndMWE(self.max_len, bert_config, self.settings.heads, self.settings.heads_mwe,
                                      self.settings.dropout)
            model.to(self.device)

            optimizer = AdamW(model.parameters(), lr=2e-5)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.settings.num_warmup_steps,
                                                        num_training_steps=self.settings.num_total_steps)

            print("fold number {}:".format(i + 1))

            scores, all_preds, all_labels, test_indices = self.trainer(self.settings.epochs, model, optimizer,
                                                                       scheduler,
                                                                       train_dataloader, test_dataloader,
                                                                       self.settings.batch_train,
                                                                       self.settings.batch_test)
            self.recorded_results_per_fold.append((scores.accuracy(),) + scores.precision_recall_fscore())

            self.all_test_indices.append(test_indices)
            self.all_predictions.append(all_preds)
            self.all_folds_labels.append(all_labels)

        return self.recorded_results_per_fold

    def prepare_data(self):
        """
            Prepares data for training.

            This involves loading data files and performing necessary preprocessing steps.

            @returns necessary vectors
        """
        df = pd.read_csv(self.settings.metaphor_dir, header=0, sep=',')
        print(f"There are {df.shape[0]} entries in the test dataset for metaphor: \n {df.head()}")
        # Create sentence and label lists
        sentences = df.sentence.values
        self.max_len = max([len(sent.split()) for sent in sentences]) + 2
        print("MAX_LEN =", self.max_len)

        language_model = SpacyModel(language_model=self.settings.language_model).get_language_model()

        # create adjacency matrix for metaphor
        A = np.array(adjacency(sentences=sentences, max_len=self.settings.max_len, language_model=language_model))

        with open(self.settings.mwe_dir + "/" + self.settings.mwe_train) as f:
            parser = CoNLLParserModel(language_model=self.settings.language_model)
            # create adjacency matrix for mwe and metaphor
            A_MWE = mwe_adjacency(f, self.settings.metaphor_dir, self.settings.max_len - 2, parser.get_parser())

        tokenized_texts = tokenize(sentences, language_model)

        # add special tokens at the beginning and end of each sentence
        for sent in tokenized_texts:
            sent.insert(0, "[CLS]")
            sent.insert(len(sent), "[SEP]")

        print("len(sentences)={}".format(len(sentences)))

        labels = df["label"].values

        target_token_indices = df["verb_idx"].values

        print("max_len of tokenized texts:", max([len(sent) for sent in tokenized_texts]))

        # construct the vocabulary
        vocab = list(set([w for sent in tokenized_texts for w in sent]))

        return A, A_MWE, labels, target_token_indices, vocab, tokenized_texts

    def prepare_bert_config(self, text, vocab):
        """
            Prepares the configuration for BERT model.

            It involves setting up model parameters, including the hidden size, number of hidden layers,
            number of attention heads, intermediate size, and the number of labels, based on the config file.

            @param text: tokenized text
            @param vocab: vocabulary

            @returns BERT config and input_ids
        """
        # index the input words
        tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-cased")
        input_ids = [tokenizer.convert_tokens_to_ids(x) for x in text]

        input_ids = pad_or_truncate(input_ids, self.max_len)

        bert_config = BertConfig(vocab_size_or_config_json_file=len(vocab))

        return input_ids, bert_config

    def train_test_loader(self, X, y, A, A_MWE, target_indices, k, batch_train, batch_test):
        """
            Generate k-fold splits given X, y, and the adjacency matrix A

            @param X: input_ids
            @param y: labels
            @param A_MWE: adjacency matrix for mwe
            @param A: adjacency matrix for metaphor
            @param target_indices: target indices
            @param k: number of folds
            @param batch_train: training set
            @param batch_test: testing set
        """
        random_state = self.random_seed
        # Create attention masks
        attention_masks = []

        # Create a mask of 1s for each token followed by 0s for padding
        for seq in X:
            seq_mask = [float(i > 0) for i in seq]
            attention_masks.append(seq_mask)
        attention_masks = np.array(attention_masks)

        kf = KFold(n_splits=k, random_state=random_state, shuffle=True)

        X = X.numpy()

        indices = np.array(target_indices)  # target token indexes
        A_MWE = np.array(A_MWE)

        for train_index, test_index in kf.split(X):
            # Extract slices
            slices = lambda data: [torch.from_numpy(data[i]) for i in [train_index, test_index]]

            X_train, X_test = slices(X)
            y_train, y_test = slices(y)
            A_train, A_test = slices(A)
            A_MWE_train, A_MWE_test = slices(A_MWE)
            indices_train, indices_test = slices(indices)
            train_masks, test_masks = slices(attention_masks)

            train_indices, test_indices = torch.tensor(train_index), torch.tensor(test_index)

            # Create TensorDatasets
            train_data = TensorDataset(X_train, train_masks, A_train, A_MWE_train, y_train, indices_train,
                                       train_indices)
            test_data = TensorDataset(X_test, test_masks, A_test, A_MWE_test, y_test, indices_test, test_indices)

            # Create DataLoaders
            train_sampler = RandomSampler(train_data)
            test_sampler = SequentialSampler(test_data)

            train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_train, drop_last=True)
            test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_test)

            yield train_dataloader, test_dataloader

    def trainer(self, epochs, model, optimizer, scheduler, train_dataloader, test_dataloader, batch_train, batch_test):
        """
            Orchestrates the training process.

            This method initializes the model, defines the optimizer and the scheduler, and controls the training loop.

            @param epochs: number of epochs
            @param model: BERT model
            @param optimizer: optimizer
            @param scheduler: optimizer scheduler
            @param train_dataloader: training set
            @param test_dataloader: testing set
            @param batch_train: batch of training set
            @param batch_test: batch of testing set

            @returns training evaluation results
        """

        max_grad_norm = self.max_grad_norm
        train_loss_set = []

        for e in trange(epochs, desc="Epoch"):

            while gc.collect() > 0:
                pass

            # Training
            # Set model to training mode (as opposed to evaluation mode)
            model.train()

            # Tracking variables
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            # Train the data for one epoch
            for step, batch in enumerate(train_dataloader):
                # Add batch to GPU
                batch = tuple(t.clone().to(torch.int64).to(self.device) for t in batch)
                # Unpack the inputs from our dataloader
                b_input_ids, b_input_mask, b_adj, b_adj_mwe, b_labels, b_target_idx, _ = batch

                # Clear out the gradients (by default they accumulate)
                optimizer.zero_grad()
                # Forward pass
                ### For BERT + GCN and MWE
                loss = model(b_input_ids.to(self.device), adj=b_adj, adj_mwe=b_adj_mwe,
                             attention_mask=b_input_mask.to(self.device), labels=b_labels,
                             batch=batch_train, target_token_idx=b_target_idx.to(self.device))

                train_loss_set.append(loss.item())
                # Backward pass
                loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                # Update parameters and take a step using the computed gradient
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # Update tracking variables
                tr_loss += loss.item()
                nb_tr_examples += b_input_ids.size(0)
                nb_tr_steps += 1

            print("Train loss: {}".format(tr_loss / nb_tr_steps))

            # Validation

            # Put model in evaluation mode to evaluate loss on the validation set
            model.eval()

            all_preds = torch.FloatTensor()
            all_labels = torch.LongTensor()
            test_indices = torch.LongTensor()

            # Evaluate data for one epoch
            for batch in test_dataloader:
                # Add batch to GPU
                batch = tuple(t.clone().to(torch.int64).to(self.device) for t in batch)
                # Unpack the inputs from our dataloader
                b_input_ids, b_input_mask, b_adj, b_adj_mwe, b_labels, b_target_idx, test_idx = batch
                # Telling the model not to compute or store gradients, saving memory and speeding up validation
                with torch.no_grad():
                    # Forward pass, calculate logit predictions
                    ### For BERT + GCN and MWE
                    logits = model(
                        b_input_ids.to(self.device),
                        adj=b_adj,
                        adj_mwe=b_adj_mwe,
                        attention_mask=b_input_mask.to(self.device),
                        batch=batch_test,
                        target_token_idx=b_target_idx.to(self.device))

                    # Move logits and labels to CPU
                    logits = logits.detach().cpu()
                    label_ids = b_labels.cpu()
                    test_idx = test_idx.cpu()

                    all_preds = torch.cat([all_preds, logits])
                    all_labels = torch.cat([all_labels, label_ids])
                    test_indices = torch.cat([test_indices, test_idx])

        scores = Evaluate(all_preds, all_labels)
        print('scores.accuracy()={}\nscores.precision_recall_fscore()={}'.format(scores.accuracy(),
                                                                                 scores.precision_recall_fscore()))

        return scores, all_preds, all_labels, test_indices
