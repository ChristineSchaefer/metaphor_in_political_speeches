import gc

import torch
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from torch.optim import AdamW
from torch.utils.data import TensorDataset, RandomSampler, DataLoader
from tqdm import trange
from transformers import BertConfig, get_linear_schedule_with_warmup, AutoTokenizer

from src.config import Settings
from src.mwe_metaphor.models.bert_model import BertWithGCNAndMWE
from src.mwe_metaphor.models.evaluation_model import Evaluate
from src.mwe_metaphor.models.spacy_model import CoNLLParserModel, SpacyModel
from src.mwe_metaphor.utils.mwe_utils import mwe_adjacency
from src.mwe_metaphor.utils.text_utils import tokenize
from src.mwe_metaphor.utils.training_utils import adjacency, pad_or_truncate

device = torch.device("mps") if torch.has_mps else torch.device("cpu")


class TrainingController(BaseModel):
    settings: Settings
    max_grad_norm: int = 1.0
    max_len: int = 0
    all_test_indices: list = Field(default_factory=list)
    all_predictions: list = Field(default_factory=list)
    all_folds_labels: list = Field(default_factory=list)
    recorded_results_per_fold: list = Field(default_factory=list)
    random_seed: int = 616

    def training(self):
        A, A_MWE, labels, target_token_indices, vocab, texts = self.prepare_data()
        input_id, bert_config = self.prepare_bert_config(texts, vocab)

        splits = self.train_test_loader(
            input_id,
            labels,
            A,
            A_MWE,
            target_token_indices,
            self.settings.K,
            self.settings.batch_train,
            self.settings.batch_test)

        for i, (train_dataloader, test_dataloader) in enumerate(splits):
            model = BertWithGCNAndMWE(self.max_len, bert_config, self.settings.heads, self.settings.heads_mwe,
                                      self.settings.dropout)
            model.to(device)

            optimizer = AdamW(model.parameters(), lr=2e-5)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.settings.num_warmup_steps,
                                                        num_training_steps=self.settings.num_total_steps)

            print('fold number {}:'.format(i + 1))

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
        df = pd.read_csv(self.settings.metaphor_dir, header=0, sep=',')
        # Create sentence and label lists
        sentences = df.sentence.values
        self.max_len = max([len(sent.split()) for sent in sentences]) + 2
        print('MAX_LEN =', self.max_len)

        A = np.array(adjacency(sentences=sentences, max_len=self.settings.max_len))

        with open(self.settings.mwe_dir + "/" + self.settings.mwe_train) as f:
            parser = CoNLLParserModel(language_model=self.settings.language_model)
            A_MWE = mwe_adjacency(f, self.settings.metaphor_dir, self.settings.max_len - 2, parser.get_parser())

        language_model = SpacyModel(language_model=self.settings.language_model).get_language_model()

        tokenized_texts = tokenize(sentences, language_model)

        # add special tokens at the beginning and end of each sentence
        for sent in tokenized_texts:
            sent.insert(0, '[CLS]')
            sent.insert(len(sent), '[SEP]')

        print('len(sentences)={}'.format(len(sentences)))

        labels = df['label'].values

        target_token_indices = df['verb_idx'].values

        print('max_len of tokenized texts:', max([len(sent) for sent in tokenized_texts]))

        # construct the vocabulary
        vocab = list(set([w for sent in tokenized_texts for w in sent]))

        return A, A_MWE, labels, target_token_indices, vocab, tokenized_texts

    def prepare_bert_config(self, text, vocab):
        # index the input words
        tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-cased")
        input_ids = [tokenizer.convert_tokens_to_ids(x) for x in text]
        # input_ids = [tokenizer(x, return_tensors="pt", padding=True)["input_ids"] for x in text]

        input_ids = pad_or_truncate(input_ids, self.max_len)

        bert_config = BertConfig(vocab_size_or_config_json_file=len(vocab))

        return input_ids, bert_config

    def train_test_loader(self, X, y, A, A_MWE, target_indices, k, batch_train, batch_test):
        """Generate k-fold splits given X, y, and the adjacency matrix A"""
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
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            A_train, A_test = A[train_index], A[test_index]
            A_MWE_train, A_MWE_test = A_MWE[train_index], A_MWE[test_index]
            indices_train, indices_test = indices[train_index], indices[test_index]  # target token indexes

            train_masks, test_masks = attention_masks[train_index], attention_masks[test_index]

            train_indices = torch.tensor(train_index)
            test_indices = torch.tensor(
                test_index)  # these are actual indices which are going to be used for retrieving items after prediction

            # Convert to torch tensors
            X_train = torch.tensor(X_train)
            X_test = torch.tensor(X_test)

            y_train = torch.tensor(y_train)
            y_test = torch.tensor(y_test)

            A_train = torch.tensor(A_train).long()
            A_test = torch.tensor(A_test).long()

            A_MWE_train = torch.tensor(A_MWE_train).long()
            A_MWE_test = torch.tensor(A_MWE_test).long()

            train_masks = torch.tensor(train_masks)
            test_masks = torch.tensor(test_masks)

            indices_train = torch.tensor(indices_train)
            indices_test = torch.tensor(indices_test)

            # Create an iterator with DataLoader
            train_data = TensorDataset(X_train, train_masks, A_train, A_MWE_train, y_train, indices_train,
                                       train_indices)
            train_sampler = RandomSampler(train_data)
            train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_train, drop_last=True)

            test_data = TensorDataset(X_test, test_masks, A_test, A_MWE_test, y_test, indices_test, test_indices)
            test_dataloader = DataLoader(test_data, sampler=None, batch_size=batch_test)

            yield train_dataloader, test_dataloader

    @staticmethod
    def trainer(epochs, model, optimizer, scheduler, train_dataloader, test_dataloader, batch_train, batch_test):

        max_grad_norm = 1.0
        train_loss_set = []

        for e in trange(epochs, desc="Epoch"):

            while gc.collect() > 0:
                pass

            # Training
            # Set our model to training mode (as opposed to evaluation mode)
            model.train()

            # if e > 8:
            #     model.freeze_bert()

            # Tracking variables
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            # Train the data for one epoch
            for step, batch in enumerate(train_dataloader):
                # Add batch to GPU
                batch = tuple(t.clone().to(torch.int64).to(device) for t in batch)
                # Unpack the inputs from our dataloader
                b_input_ids, b_input_mask, b_adj, b_adj_mwe, b_labels, b_target_idx, _ = batch

                # Clear out the gradients (by default they accumulate)
                optimizer.zero_grad()
                # Forward pass
                ### For BERT + GCN and MWE
                loss = model(b_input_ids.to(device), adj=b_adj, adj_mwe=b_adj_mwe,
                             attention_mask=b_input_mask.to(device), labels=b_labels,
                             batch=batch_train, target_token_idx=b_target_idx.to(device))

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
                batch = tuple(t.clone().to(torch.int64).to(device) for t in batch)
                # Unpack the inputs from our dataloader
                b_input_ids, b_input_mask, b_adj, b_adj_mwe, b_labels, b_target_idx, test_idx = batch
                # Telling the model not to compute or store gradients, saving memory and speeding up validation
                with torch.no_grad():
                    # Forward pass, calculate logit predictions
                    ### For BERT + GCN and MWE
                    logits = model(b_input_ids.to(device), adj=b_adj, adj_mwe=b_adj_mwe,
                                   attention_mask=b_input_mask.to(device), \
                                   batch=batch_test, target_token_idx=b_target_idx.to(device))

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
