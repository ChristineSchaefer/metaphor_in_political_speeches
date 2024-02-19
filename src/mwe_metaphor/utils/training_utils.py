import itertools

import pandas as pd
import torch
import numpy as np

from src.mwe_metaphor.utils.tsvlib import iter_tsv_sentences

# from https://github.com/omidrohanian/metaphor_mwe/blob/master/utils.py
# and https://github.com/omidrohanian/metaphor_mwe/blob/master/mwe/myMWEProcess.py


def adjacency(sentences, max_len, language_model):
    """
        Compute dependent-to-head adjacency matrices.

        @param sentences: list of sentences
        @param max_len: maximum of sentence length
        @param language_model: spacy language model

        @returns adjacency matrix of shape (max_len, max_len)
    """

    A = []
    for sent in sentences:
        doc = language_model(sent)
        adj = np.zeros([max_len, max_len])
        for tok in doc:
            if not str(tok).isspace():
                if tok.i + 1 < max_len and tok.head.i + 1 < max_len:
                    adj[tok.i + 1][tok.head.i + 1] = 1
        A.append(adj)
    return A


def pad_or_truncate(input_ids, max_len):
    """
        Pads or truncates a list to a target length.

        @param input_ids: list of input_ids
        @param max_len: maximum of sentence length

        @returns padded or truncated list of input_ids
    """
    pad = lambda seq, max_len: seq[0:max_len] if len(seq) > max_len else seq + [0] * (max_len - len(seq))
    return torch.Tensor([pad(seq, max_len) for seq in input_ids]).long()


def mwe_adjacency(input_file_obj, file_dir, max_len, parser):
    """
        Returns a list, which contains adjacency matrices for MWEs in the input sentences.

        @param input_file_obj: file to parseme dataset with MWEs
        @param file_dir: file to metaphor dataset
        @param max_len: maximum of sentence length
        @param parser: parser

        @returns adjacency matrix for mwe of shape (max_len, max_len)
    """
    mwe_adj = []

    # list with TSVSentence objects from parseme data set
    parseme_sents = list(iter_tsv_sentences(input_file_obj))
    print("len of sents in mwe adj processing", len(list(parseme_sents)))

    df = pd.read_csv(file_dir, header=0, sep=None, engine="python")
    sentences = df.sentence.values
    i = 0
    for sent in sentences:
        # initializes a square matrix a of zeros with dimensions (max_len, max_len)
        # to represent the adjacency matrix for the current sentence
        a = np.zeros((max_len, max_len), dtype=int)
        pivot = 0
        for subsent in parser.parse_text_as_conll(text=sent.lstrip()):
            if i > len(parseme_sents):
                print(i)
            s = parseme_sents[i]
            # or if you want the length of the sentence, it is len(s.words)
            for mwe in s.mwe_infos():
                # calculates the position b of the first token in the MWE within the adjacency matrix
                b = s.mwe_infos()[mwe].token_indexes[0] + pivot
                # computes combinations of token indexes within the MWE
                comb = itertools.combinations(s.mwe_infos()[mwe].token_indexes, 2)
                for i, j in comb:  # s.mwe_infos()[mwe].token_indexes[1:]:
                    if j + pivot < max_len and i + pivot < max_len:
                        # sets the corresponding entries in the adjacency matrix a to 1 based on the token indexes,
                        # indicating a connection between the tokens within the MWE
                        a[j + pivot][i + pivot] = 1
                        a[i + pivot][j + pivot] = 1
                        b = i + pivot
            pivot = pivot + len(s.words)

        a = np.array(a)
        a = np.concatenate((np.zeros((1, max_len), dtype=int), a,
                            np.zeros((1, max_len), dtype=int)), axis=0)
        a = np.concatenate((np.zeros((a.shape[0], 1), dtype=int), a,
                            np.zeros((a.shape[0], 1), dtype=int)), axis=1)
        mwe_adj.append(a)
        i += 1
    return mwe_adj
