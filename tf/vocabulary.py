from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Counter, OrderedDict

import numpy as np

import tensorflow as tf
import re
import os

from tensorflow.gfile import Open as open
from tensorflow.gfile import Exists as exists


class Vocab(object):
    def __init__(self, special=[], min_freq=0, max_size=None, lower_case=True,delimiter=None, vocab_file=None):
        self.counter = Counter()
        self.special = special
        self.min_freq = min_freq
        self.max_size = max_size
        self.lower_case = lower_case
        self.delimiter = delimiter
        self.vocab_file = vocab_file
        self.idx2sym = []
        self.sym2idx = OrderedDict()

    ### LAI: Simple sentence split based on punctuation
    def sent_token_split(self, doc):
        doc_modified = re.sub(r' ', "", doc)

        doc_modified = re.sub(r'。', r"。 ", doc_modified)
        doc_modified = re.sub(r'！', r"！ ", doc_modified)
        doc_modified = re.sub(r'？', r"？ ", doc_modified)
        doc_modified = re.sub('\t', " ", doc_modified)
    
        doc_split = re.split(r' ', doc_modified)

        doc_split = [i for i in doc_split if len(i.strip())>0]
    
        return doc_split

    def tokenize(self, line, add_eos=False, add_double_eos=False):
        line = line.strip()

        if add_double_eos:
            symbols = list(line)
            self.add_symbol('<S>')
            return ['<S>'] + symbols + ['<S>']
        elif add_eos:
            """
            symbols = []
            for i in self.sent_token_split(line):
                sent = list(i)
                sent = ['<bos>'] + sent + ['<eos>']
                symbols.extend(sent)
            """
            symbols = list(line)

            return ['<bos>'] + symbols + ['<eos>']
        else:
            symbols = list(line)
            return symbols

    # Extract sentence from file
    def count_file(self, path, verbose=False, add_eos=False):
        if verbose: print('counting file {} ...'.format(path))
        assert exists(path)

        sents = []
        with open(path, 'r') as f:
            for idx, line in enumerate(f):
                if verbose and idx > 0 and idx % 500000 == 0:
                    print('  line {}'.format(idx))
                symbols = self.tokenize(line, add_eos=add_eos)
                self.counter.update(symbols)
                sents.append(symbols)

        return sents

    def count_sents(self, sents, verbose=False):
        """
          sents : a list of sentences, each a list of tokenized symbols
        """
        if verbose: print('counting {} sents ...'.format(len(sents)))
        for idx, symbols in enumerate(sents):
            if verbose and idx > 0 and idx % 500000 == 0:
                print('  line {}'.format(idx))
            self.counter.update(symbols)

    def _build_from_file(self, vocab_file):
        self.idx2sym = []
        self.sym2idx = OrderedDict()

        with open(vocab_file, 'r') as f:
            for line in f:
                symb = line.replace("\n","")
                self.add_symbol(symb)
        
        self.unk_idx = self.sym2idx['<UNK>']

    def build_vocab(self):
        if self.vocab_file:
            print('building vocab from {}'.format(self.vocab_file))
            self._build_from_file(self.vocab_file)
            print('final vocab size {}'.format(len(self)))

            self.unk_idx = self.sym2idx['<UNK>']
        else:
            print('building vocab with min_freq={}, max_size={}'.format(
                self.min_freq, self.max_size))

            for i in self.special:
                self.add_special(i)
            self.unk_idx = self.sym2idx['<UNK>']

            tmp = sorted(self.counter.items(), key=lambda item:item[1], reverse=True)

            if self.max_size == None:
                self.max_size = len(tmp)
            for i in range(len(tmp)):
                if i+1>=self.max_size or tmp[i][1]<=self.min_freq or i+1==len(tmp):
                    cut = i+1
                    break
            tmp = tmp[:cut]
            tmp.sort(key=lambda x: x[0])

            for sym, _ in tmp:
                self.add_symbol(sym)

            print('final vocab size {} from {} unique tokens'.format(
                len(self), len(self.counter)))

    def save_vocab(self, path, vocab_name):
        with open(os.path.join(path, "vocabulary_%s.txt" % vocab_name), "w") as write_vocab:
            for i in self.idx2sym:
                write_vocab.write(i+"\n")
        
        print("  Vocabulary of {} is saved under path: {}".format(vocab_name, os.path.abspath(path)))   

    def encode_file(self, path, ordered=False, verbose=False, add_eos=False,
                    add_double_eos=False):
        if verbose: print('encoding file {} ...'.format(path))
        assert exists(path)
        encoded = []
        with open(path, 'r') as f:
            for idx, line in enumerate(f):
                if verbose and idx > 0 and idx % 500000 == 0:
                    print('  line {}'.format(idx))
                if len(line.strip()) == 0:
                    continue
                symbols = self.tokenize(line, add_eos=add_eos, add_double_eos=add_double_eos)

                encoded.append(self.convert_to_nparray(symbols))

        if ordered:
            encoded = np.concatenate(encoded)

        return encoded

    def encode_sents(self, sents, add_eos=False, ordered=False, verbose=False):
        if verbose: print('encoding {} sents ...'.format(len(sents)))
        encoded = []

        symbols = self.tokenize(sents, add_eos=add_eos)
        encoded.append(self.convert_to_nparray(symbols))

        if ordered:
            encoded = np.concatenate(encoded)

        return encoded

    def add_special(self, sym):
        if sym not in self.sym2idx:
            self.idx2sym.append(sym)
            self.sym2idx[sym] = len(self.idx2sym) - 1
            setattr(self, '{}_idx'.format(sym.strip('<>')), self.sym2idx[sym])

    def add_symbol(self, sym):
        if sym not in self.sym2idx:
            self.idx2sym.append(sym)
            self.sym2idx[sym] = len(self.idx2sym) - 1

    def get_sym(self, idx):
        assert 0 <= idx < len(self.idx2sym), 'Index {} out of range'.format(idx)
        return self.idx2sym[idx]

    def get_idx(self, sym):
        if sym in self.sym2idx:
            return self.sym2idx[sym]
        else:
            assert hasattr(self, 'unk_idx')
            return self.sym2idx.get(sym, self.unk_idx)

    def get_symbols(self, indices):
        return [self.get_sym(idx) for idx in indices]

    def get_indices(self, symbols):
        return [self.get_idx(sym) for sym in symbols]

    ### symbol to index
    def convert_to_nparray(self, symbols):
        nparray = np.array(self.get_indices(symbols), dtype=np.int64)
        return nparray

    ### index to symbol
    def convert_to_sent(self, indices, exclude=None):
        if exclude is None:
            return ' '.join([self.get_sym(idx) for idx in indices])
        else:
            return ' '.join([self.get_sym(idx) for idx in indices if idx not in exclude])

    def __len__(self):
        return len(self.idx2sym)
