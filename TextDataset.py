import torch
import os
import json
import io
import numpy as np
from collections import defaultdict
from nltk.tokenize import TweetTokenizer
from torch.utils.data import Dataset
from utils import OrderedCounter

class TextDataset(Dataset):

    def __init__(self, root, split, pre, **kwargs):

        super().__init__()

        self.root = root
        self.pre = pre

        assert split in ['train', 'valid', 'test'], "Split must be one of: 'train', 'valid', 'test'. But received %s"%(split)
        self.split = split

        self.max_sequence_length = kwargs.get('max_sequence_length', 40)
        self.min_occ = kwargs.get('min_occ', 10)

        self._create_text_dataset(**kwargs)

    def __getitem__(self, idx):

        return {
            'words': self.data[idx]['words'],
            'inputs': np.asarray(self.data[idx]['inputs']),
            'targets': np.asarray(self.data[idx]['targets']),
            'len': self.data[idx]['len']
        }
    def __len__(self):
        return len(self.data)

    @property
    def vocab_size(self):
        return len(self.w2i)

    @property
    def pad_token(self):
        return self.w2i['<pad>']

    def _create_text_dataset(self, **kwargs):

        dataset_raw_file = os.path.join(self.root, self.pre + '.' + self.split + '.txt')
        assert os.path.exists(dataset_raw_file), "File %s not found."%(dataset_raw_file)

        self.data = defaultdict(dict)

        if self.split == 'train':
            self._create_vocab(dataset_raw_file, **kwargs)
        else:
            self._load_vocab()

        tokenizer = TweetTokenizer(preserve_case=False)
        with open(dataset_raw_file, 'r') as file:

            for sentence in file.readlines():

                # make list of words from sentence
                words = tokenizer.tokenize(sentence)

                id = len(self.data)

                # add sos token
                self.data[id]['words'] = ['<sos>'] + words

                # cut off if exceeds max_sequence_length
                self.data[id]['words'] = self.data[id]['words'][:self.max_sequence_length-1] # -1 to make space for <eos> token

                # add <eos> token
                self.data[id]['words'] += ['<eos>']

                # save length before padding
                self.data[id]['len'] = len(self.data[id]['words']) - 1

                # pad
                self.data[id]['words'].extend(['<pad>'] * (self.max_sequence_length - len(self.data[id]['words'])))

                # convert to idicies
                word_idx = [self.w2i[w] if w in self.w2i else self.w2i['<unk>'] for w in self.data[id]['words']]

                # create inputs and targets
                self.data[id]['inputs'] = word_idx[:-1]
                self.data[id]['targets'] = word_idx[1:]

                assert self.data[id]['len'] <= self.max_sequence_length


        print("%s dataset created with %i dataponts."%(self.split, len(self.data)))



    def _create_vocab(self, dataset_raw_file, **kwargs):

        assert self.split == 'train', "Vocablurary can only be created from training file."

        tokenizer = TweetTokenizer(preserve_case=False)

        w2c = OrderedCounter()
        w2i = dict()
        i2w = dict()

        # add speical tokens to vocab
        special_tokens = ['<pad>', '<sos>', '<eos>', '<unk>']
        for st in special_tokens:
            i2w[len(w2i)] = st
            w2i[st] = len(w2i)

        with open(dataset_raw_file, 'r') as file:

            # read data and count token occurences
            for line in file.readlines():
                tokens = tokenizer.tokenize(line)
                w2c.update(tokens)

            # create vocab with
            for w, c in w2c.items():
                if c > self.min_occ:
                    i2w[len(w2i)] = w
                    w2i[w] = len(w2i)

        vocab = dict(w2i=w2i, i2w=i2w)

        # save vocab to file
        vocab_file_path = os.path.join(self.root, 'vocab.json')
        with io.open(vocab_file_path, 'wb') as vocab_file:
            data = json.dumps(vocab, ensure_ascii=False)
            vocab_file.write(data.encode('utf8', 'replace'))

        print("Vocablurary created with %i tokens. Minimum occurence criterion = %i."%(len(w2i), self.min_occ))

        self._load_vocab()

    def _load_vocab(self):

        vocab_file_path = os.path.join(self.root, 'vocab.json')
        assert os.path.exists(vocab_file_path), "Vocablurary file at %s not found."%(vocab_file_path)

        with open(vocab_file_path, 'r') as vocab_file:
            vocab = json.load(vocab_file)

        self.w2i, self.i2w = vocab['w2i'], vocab['i2w']
