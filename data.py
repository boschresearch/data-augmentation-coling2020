"""
Code for simple data augmentation methods for named entity recognition (Coling 2020).
Copyright (c) 2020 - for information on the respective copyright owner see the NOTICE file.

SPDX-License-Identifier: Apache-2.0

The code in this file is partly based on the FLAIR library,
(https://github.com/flairNLP/flair), licensed under the MIT license,
cf. 3rd-party-licenses.txt file in the root directory of this source tree.
"""

import json, logging, torch

logger = logging.getLogger(__name__)

class Dictionary:
    def __init__(self, unk_value="<unk>"):
        self.item2idx = {}
        self.idx2item = []
        self.unk_idx = self.add_item(unk_value) if unk_value is not None else None

    def add_item(self, item):
        # item = item.encode("utf-8")
        if item not in self.item2idx:
            self.item2idx[item] = len(self.idx2item)
            self.idx2item.append(item)
        return self.item2idx[item]

    def get_idx(self, item):
        return self.item2idx.get(item, self.unk_idx)

    def get_item(self, idx):
        # return self.idx2item[idx].decode("UTF-8")
        return self.idx2item[idx]

    def __len__(self):
        return len(self.idx2item)

    def __str__(self):
        return json.dumps(self.item2idx)

    def __repr__(self):
        return self.__str__()


class Token:
    def __init__(self, text, idx=None):
        super(Token, self).__init__()
        self.text = text
        self.idx = idx                                          # index in the sentence
        self.num_subtokens = None                               # how many sub tokens the original token is splitted
        self._embeddings = {}
        self._labels = {}                                       # key could be 'gold', 'pred'

    def set_embedding(self, name, vector, device):
        if vector.device != device: vector = vector.to(device)
        self._embeddings[name] = vector

    def to(self, device, pin_memory=False):
        for k, v in self._embeddings.items():
            if v.device != device:
                if pin_memory:
                    self._embeddings[k] = v.to(device, non_blocking=True).pin_memory()
                else:
                    self._embeddings[k] = v.to(device,non_blocking=True)

    def clear_embeddings(self, embedding_names=None):
        if embedding_names is None:
            self._embeddings = {}
        else:
            for name in embedding_names:
                if name in self._embeddings.keys():
                    del self._embeddings[name]

    def get_embedding_list(self, device):
        return [self._embeddings[k].to(device) for k in sorted(self._embeddings.keys())]

    def get_embedding(self):
        return torch.cat(self.get_embedding_list(), dim=0)

    def set_label(self, label_type, label_value):
        self._labels[label_type] = label_value

    def get_label(self, label_type):
        return self._labels[label_type]

    def __str__(self):
        return self.text

    def __repr__(self):
        return self.__str__()


class Span:
    def __init__(self, tokens, label):
        self.tokens = tokens
        self.label = label

    @property
    def text(self):
        return " ".join([t.text for t in self.tokens])

    def __str__(self) -> str:
        ids = ",".join([str(t.idx) for t in self.tokens])
        return "%s-span [%s]: %s" % (self.label, ids, self.text)

    def __repr__(self) -> str:
        return self.__str__()


class Sentence:
    def __init__(self, idx):
        super(Sentence, self).__init__()
        self.idx = idx                                              # index in the dataset
        self.tokens = []
        self.tokens_indices = None                                  # a sequence of sub token IDs
        self._embeddings = {}

    def get_token(self, token_idx):
        for token in self.tokens:
            if token.idx == token_idx:
                return token

    def add_token(self, token):
        if type(token) is str: token = Token(token)
        if token.idx is None: token.idx = len(self.tokens)
        self.tokens.append(token)

    def to(self, device, pin_memory=False):
        for k, v in self._embeddings.items():
            if v.device != device:
                if pin_memory:
                    self._embeddings[k] = v.to(device, non_blocking=True).pin_memory()
                else:
                    self._embeddings[k] = v.to(device, non_blocking=True)
        for t in self:
            t.to(device, pin_memory)

    def clear_embeddings(self, embedding_names=None):
        if embedding_names is None:
            self._embeddings = {}
        else:
            for name in embedding_names:
                if name in self._embeddings.keys():
                    del self._embeddings[name]
        for t in self:
            t.clear_embeddings(embedding_names)

    def __iter__(self):
        return iter(self.tokens)

    def __getitem__(self, idx):
        return self.tokens[idx]

    def __len__(self):
        return len(self.tokens)

    def __str__(self):
        return "%s: %s" % (self.idx, " ".join([t.text for t in self]))

    def __repr__(self):
        return self.__str__()


def get_spans(sentence, label_type):
    spans = []
    tokens_in_span = []
    prev_label = "O"
    for token in sentence:
        label = token.get_label(label_type)
        in_span, starts_span = False, False
        if label != "O":
            in_span = True
            assert label[0] in ["B", "I", "E", "S"]
            starts_span = (label[0] in ["B", "S"])
            if (prev_label == "O" or prev_label[0] in ["E", "S"]): starts_span = True
            if prev_label[2:] != label[2:]: starts_span = True

        if (starts_span or not in_span) and len(tokens_in_span) > 0:
            spans.append(Span(tokens_in_span, prev_label[2:]))
            tokens_in_span = []
        if in_span:
            tokens_in_span.append(token)
        prev_label = label
    if len(tokens_in_span) > 0: spans.append(Span(tokens_in_span, prev_label[2:]))

    return spans


class ConllDataset(torch.utils.data.Dataset):
    def __init__(self, name, filepath=None):
        self.name = name
        self.sentences = []

        if filepath is not None:
            idx = 0
            tokens, tags = [], []
            with open(filepath, encoding="utf-8") as f:
                for line in f:
                    if line.isspace():
                        if len(tokens) > 0:
                            self.sentences.append(ConllDataset.create_sentence("%s-%d" % (name, idx), tokens, tags))
                            idx += 1
                        tokens, tags = [], []
                    else:
                        sp = line.strip().split()
                        assert len(sp) == 2
                        tokens.append(sp[0])
                        tags.append(sp[1])
                if len(tokens) > 0:
                    self.sentences.append(ConllDataset.create_sentence("%s-%d" % (name, idx), tokens, tags))
                    idx += 1
            assert idx == len(self.sentences)
            logger.info("Load %s sentences from %s" % (len(self.sentences), filepath))

    @staticmethod
    def create_sentence(idx, tokens, tags, label_type="gold"):
        sentence = Sentence(idx=idx)
        for t, tag in zip(tokens, tags):
            token = Token(t)
            token.set_label(label_type, tag)
            sentence.add_token(token)
        return sentence

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]


class Corpus:
    def __init__(self, train, dev=None, test=None, unlabel=None, name="corpus"):
        self.train = train
        self.dev = dev
        self.test = test
        self.unlabel = unlabel
        self.name = name


class ConllCorpus(Corpus):
    def __init__(self, name, train_filepath, dev_filepath=None, test_filepath=None):
        train = ConllDataset("%s-train" % name, train_filepath)
        dev = ConllDataset("%s-dev" % name, dev_filepath) if dev_filepath is not None else None
        test = ConllDataset("%s-test" % name, test_filepath) if test_filepath is not None else None
        super(ConllCorpus, self).__init__(train, dev, test, name=name)

    def build_tag_dict(self, label_type):
        sentences = [self.train]
        if self.dev is not None: sentences = sentences + [self.dev]
        if self.test is not None: sentences = sentences + [self.test]
        sentences = torch.utils.data.dataset.ConcatDataset(sentences)

        dict = Dictionary(unk_value=None)
        dict.add_item("O")
        for s in sentences:
            for t in s:
                dict.add_item(t.get_label(label_type))
        return dict