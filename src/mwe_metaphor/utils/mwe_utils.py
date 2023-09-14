import numpy as np
import pandas as pd
import itertools

from src.mwe_metaphor.utils.tsvlib import iter_tsv_sentences


def mwe_adjacency(input_file_obj, file_dir, max_len, parser):
    """
    Returns a list, which contains adjacency matrices for MWEs in the input sentences.
    :param input_file_obj: file to parseme dataset with MWEs
    :param file_dir: file to metaphor dataset
    :param max_len:
    :return:
    """
    mwe_adj = []

    # list with TSVSentence objects from parseme data set
    parseme_sents = list(iter_tsv_sentences(input_file_obj))
    print('len of sents in mwe adj processing', len(list(parseme_sents)))

    df = pd.read_csv(file_dir, header=0, sep=None, engine='python')
    sentences = df.sentence.values
    i = 0
    for sent in sentences:
        # initializes a square matrix a of zeros with dimensions (max_len, max_len)
        # to represent the adjacency matrix for the current sentence
        a = np.zeros((max_len, max_len), dtype=int)
        # doc = nlp(sent)
        pivot = 0
        for subsent in parser.parse_text_as_conll(text=sent.lstrip()):
        # for subsent in doc.sents:
            # try:
            #    s = next(parseme_sents)
            # except StopIteration:
            #    break
            # s = next(parseme_sents)
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
            # if pivot>0 and len(s.mwe_infos()):
            #   print(sent)
            #   print(s.mwe_infos())
            #   print(a[pivot:][pivot:])
            pivot = pivot + len(s.words)

        a = np.array(a)
        a = np.concatenate((np.zeros((1, max_len), dtype=int), a,
                            np.zeros((1, max_len), dtype=int)), axis=0)
        a = np.concatenate((np.zeros((a.shape[0], 1), dtype=int), a,
                            np.zeros((a.shape[0], 1), dtype=int)), axis=1)
        # print(a.shape)
        mwe_adj.append(a)
        i += 1
    # print('mwe adj matrix sample:', [i for a in mwe_adj[0:29] for i in a if 1 in i])
    return mwe_adj


def count_mwes(mwe_file, metaphor_file):
    # This is for saif MOH dataset which is already tokenized
    # and has only one sentence in each entry of data
    df = pd.read_csv(metaphor_file, header=0, sep=',')
    sentences = df.sentence.values
    verb_idxes = df.verb_idx.values
    labels = df.label.values

    with open(mwe_file) as f:
        parseme_sents = [s for s in iter_tsv_sentences(f)]

    print("metaphor data len:", len(sentences), "mwe data len:", len(parseme_sents))

    metaphor_count = metaphorMWE_count = metaphor_MWEinSent = verbal_mwe = 0
    mwe_target_verb = []
    for i in range(len(sentences)):
        sent = sentences[i].strip().split(' ')
        if sent[0] == '':
            sent = sent[1:]
        ps = parseme_sents[i].words
        assert len(sent) == len(ps), "Number of words in this sentence do not match:" + str(sent) + str(
            len(sent)) + " " + str(len(ps))

        verb_idx = verb_idxes[i]
        verb_is_mwe = 0
        for mwe in parseme_sents[i].mwe_infos():
            if verb_idx in parseme_sents[i].mwe_infos()[mwe].token_indexes:
                mwe_target_verb.append(i)
                verb_is_mwe = 1
        if labels[i] == 1 and verb_is_mwe:
            metaphorMWE_count += 1
        if labels[i] == 1 and parseme_sents[i].mwe_infos():
            metaphor_MWEinSent += 1

        if labels[i] == 1:
            metaphor_count += 1

        else:
            verb_is_mwe = 0
            for mwe in parseme_sents[i].mwe_infos():
                if verb_idx in parseme_sents[i].mwe_infos()[mwe].token_indexes:
                    verb_is_mwe = 1
                    print("!!! Non-metaphor verb, *{}*".format(sent[verb_idx]), "IN",
                          sentences[i])  # , 'IS ANNOTATED AS MWE:', parseme_sents[i].mwe_infos()[mwe])
            if verb_is_mwe == 1:
                verbal_mwe += 1

    print('From', metaphor_count, 'metaphors', metaphorMWE_count, 'are part of MWEs')
    print('But', verbal_mwe, 'mwe target verbs are not metaphor')
    print('From', metaphor_count, 'metaphors', metaphor_MWEinSent, 'have some kind of MWEs in the sentences')
    print(mwe_target_verb)