#! /usr/bin/env python3
# from https://github.com/omidrohanian/metaphor_mwe/blob/master/mwe/tsvlib.py

"""
    This is a small library for reading and interpreting
    the new ConLLU-PLUS format.

    This format allows any column from CoNLLU (e.g. ID, FORM...)
    As in CoNLL-U, empty columns are represented by "_".

    The first line of these files should have the form:
    # global.columns = ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC PARSEME:MWE

    The column "PARSEME:MWE" can be used to indicate
    MWE codes (e.g. "3:LVC.full;2;5:VID") or be EMPTY.
"""

import collections
import os
import sys

from pydantic import ConfigDict, Field

from src.database import Document

UNDERSP = "_"
SINGLEWORD = "*"


#######################################
def interpret_color_request(stream, color_req: str) -> bool:
    """
        Interpret environment variables COLOR_STDOUT and COLOR_STDERR ("always/never/auto").

        @param stream
        @param color_req

        @returns bool
    """
    return color_req == 'always' or (color_req == 'auto' and stream.isatty())


# Flags indicating whether we want to use colors when writing to stderr/stdout
COLOR_STDOUT = interpret_color_request(sys.stdout, os.getenv('COLOR_STDOUT', 'auto'))
COLOR_STDERR = interpret_color_request(sys.stderr, os.getenv('COLOR_STDERR', 'auto'))


############################################################
class TSVSentence(Document):
    """
        A list of TSVTokens.
        TSVTokens may include ranges and sub-tokens.

        For example, if we have these TSVTokens:
            1   You
            2-3 didn't   -- a range
            2   did      -- a sub-token
            3   not      -- a sub-token
            4   go
        Iterating through `self.words` will yield ["You", "did", "not", "go"].
        You can access the range ["didn't"] through `self.contractions`.
    """
    filename: str = Field(..., description="filename of the TSV file")
    words: list | None = Field(default_factory=list, description="list of words")
    contractions: list | None = Field(default_factory=list, description="list of contractions")
    lineno_bag: int | None = Field(default=None,description="line with no bag")
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    class Settings:
        collection_name = "tsv_sentences"

    def append(self, token):
        """
            Add `token` to either `self.words` or `self.contractions`.

            @param token
        """
        L = (self.contractions if token.is_contraction() else self.words)
        L.append(token)

    def mwe_infos(self):
        """
            Return a dict {mwe_id: MWEInfo} for all MWEs in this sentence.

            @returns dict with mwe info
        """
        mwe_infos = {}
        for token_index, token in enumerate(self.words):
            global_last_lineno(self.filename, token.lineno)
            for mwe_id, mwe_categ in token.mwes_id_categ():
                mwe_info = mwe_infos.setdefault(mwe_id, MWEInfo(self, mwe_categ, []))
                mwe_info.token_indexes.append(token_index)
        return mwe_infos


class MWEInfo(collections.namedtuple('MWEInfo', 'sentence category token_indexes')):
    """
        Represents a single MWE in a sentence.
        CAREFUL: token indexes start at 0 (not at 1, as in the TokenID's).
    """

    def n_tokens(self):
        """
            @returns the number of tokens in self
        """
        return len(self.token_indexes)


class TSVToken(collections.UserDict):
    """
        Represents a token in the TSV file.
        You can index this object to get the value of a given field
        (e.g. self["FORM"] or self["PARSEME:MWE"]).
    """

    def __init__(self, lineno, data):
        """
            Initialize the TSVToken class.
        """
        self.lineno = lineno
        super().__init__(data)

    def mwe_codes(self):
        """
            @returns a set of MWE codes
        """
        mwes = self['PARSEME:MWE']
        return set(mwes.split(';') if mwes != SINGLEWORD else ())

    def mwes_id_categ(self):
        """
            For each MWE code in `self.mwe_codes`, yield an (id, categ) pair.

            @rtype Iterable[(int, Optional[str])]
        """
        for mwe_code in sorted(self.mwe_codes()):
            yield mwe_code_to_id_categ(mwe_code)

    def is_contraction(self):
        """
            Return True iff this token represents a range of tokens.
            (The following tokens in the TSVSentence will contain its elements).

            @returns true if token represents a range of token
        """
        return "-" in self.get('ID', '')

    def __missing__(self, key):
        """
            Override _missing_ method
        """
        raise KeyError('''Field {} is underspecified ("_" or missing)'''.format(key))


def mwe_code_to_id_categ(mwe_code):
    """
        Convert mwe_code_to_id_categ(mwe_code) to (mwe_id, mwe_categ).

        @param mwe_code: mwe_code to convert

        @returns mwe_id, mwe_categ
    """
    split = mwe_code.split(":")
    mwe_id = int(split[0])
    mwe_categ = (split[1] if len(split) > 1 else None)
    return mwe_id, mwe_categ


############################################################


def iter_tsv_sentences(fileobj):
    """
        Yield `TSVSentence` instances for all sentences in the underlying PARSEME TSV file.

        @param fileobj: file
    """
    header = next(fileobj)
    if not 'global.columns' in header:
        exit('ERROR: {}: file is not in the required format: missing global.columns header' \
             .format(os.path.basename(fileobj.name) if len(fileobj.name) > 30 else fileobj.name))
    colnames = header.split('=')[-1].split()

    sentence = None
    for lineno, line in enumerate(fileobj, 2):
        global_last_lineno(fileobj.name, lineno)
        if line.startswith("#"):
            pass  # Skip comments
        elif line.strip():
            if not sentence:
                sentence = TSVSentence(filename=fileobj.name, lineno_bag=lineno)
            fields = line.strip().split('\t')
            if len(fields) != len(colnames):
                raise Exception('Line has {} columns, but header specifies {}' \
                                .format(len(fields), len(colnames)))
            data = {c: f for (c, f) in zip(colnames, fields) if f != UNDERSP}
            sentence.append(TSVToken(lineno, data))
        else:
            if sentence:
                yield sentence
                sentence = None
    if sentence:
        yield sentence


####################################################################

last_filename = None
last_lineno = 0


def global_last_lineno(filename, lineno):
    """
       Update global `last_lineno` var.

       @param filename
       @param lineno
    """
    global last_filename
    global last_lineno
    last_filename = filename
    last_lineno = lineno


_MAX_WARNINGS = 10
_WARNED = collections.defaultdict(int)


def warn(message, *, warntype="WARNING", position=None, **format_args):
    """
        Warning method.
    """
    _WARNED[message] += 1
    if _WARNED[message] <= _MAX_WARNINGS:
        if position is None:
            position = "{}:{}: ".format(last_filename, last_lineno) if last_filename else ""
        msg_list = message.format(**format_args).split("\n")
        if _WARNED[message] == _MAX_WARNINGS:
            msg_list.append("(Skipping following warnings of this type)")

        line_beg, line_end = ('\x1b[31m', '\x1b[m') if COLOR_STDERR else ('', '')
        for i, msg in enumerate(msg_list):
            warn = warntype if i == 0 else "." * len(warntype)
            print(line_beg, position, warn, ": ", msg, line_end, sep="", file=sys.stderr)


def excepthook(exctype, value, tb):
    """
        Exception handler.
    """
    global last_lineno
    global last_filename
    if value and last_lineno:
        last_filename = last_filename or "???"
        err_msg = "===> ERROR when reading {} (line {})" \
            .format(last_filename, last_lineno)
        if COLOR_STDERR:
            err_msg = "\x1b[31m{}\x1b[m".format(err_msg)
        print(err_msg, file=sys.stderr)
    return sys.__excepthook__(exctype, value, tb)


#####################################################################
