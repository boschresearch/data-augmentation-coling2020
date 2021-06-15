"""
Code for simple data augmentation methods for named entity recognition (Coling 2020).
Copyright (c) 2020 - for information on the respective copyright owner see the NOTICE file.

SPDX-License-Identifier: Apache-2.0
"""

import random
from collections import defaultdict
import numpy as np
import nltk
STOPWORDS = set(nltk.corpus.stopwords.words("english"))
from nltk.corpus import wordnet
from data import Sentence, Token


def get_category2mentions(dataset):
    mentions = []
    for sentence in dataset:
        mention = []
        for token in sentence:
            label = token.get_label("gold")
            if label == "O" or label[0] == "B":
                if len(mention) > 0:
                    mentions.append(mention)
                mention = []
            if label[0] == "B": mention.append(label[2:])
            if label != "O": mention.append(token.text)
        if len(mention) > 0:
            mentions.append(mention)

    category2mentions = {}
    for mention in mentions:
        if mention[0] not in category2mentions: category2mentions[mention[0]] = {}
        category2mentions[mention[0]][" ".join(mention[1:])] = 1

    for category in category2mentions.keys():
        mentions = list(category2mentions[category].keys())
        category2mentions[category] = mentions
    return category2mentions


def generate_sentences_by_replace_mention(sentence, category2mentions, replace_ratio, num_generated_samples):
    generated_sentences = []
    for i in range(num_generated_samples):
        generated_sentence = Sentence("%s-replace-mention-%d" % (sentence.idx, i))
        for j, token in enumerate(sentence.tokens):
            label = token.get_label("gold")
            if label == "O":
                generated_token = Token(token.text)
                generated_token.set_label("gold", label)
                generated_sentence.add_token(generated_token)
            elif label[0] == "B":
                category = label[2:]
                if np.random.binomial(1, replace_ratio, 1)[0]:
                    candidates = category2mentions[category]
                    random_idx = np.random.choice(len(candidates), 1)[0]
                    replaced_mention = candidates[random_idx].split()
                    generated_token = Token(replaced_mention[0])
                    generated_token.set_label("gold", "B-%s" % category)
                    generated_sentence.add_token(generated_token)
                    for t in replaced_mention[1:]:
                        generated_token = Token(t)
                        generated_token.set_label("gold", "I-%s" % category)
                        generated_sentence.add_token(generated_token)
                else:
                    generated_token = Token(token.text)
                    generated_token.set_label("gold", "B-%s" % category)
                    generated_sentence.add_token(generated_token)
                    next = j + 1
                    while next < len(sentence) and sentence[next].get_label("gold")[0] == "I":
                        next_token = sentence[next]
                        generated_token = Token(next_token.text)
                        generated_token.set_label("gold", "I-%s" % category)
                        generated_sentence.add_token(generated_token)
                        next += 1
            elif label[0] == "I":
                continue
            else:
                raise ValueError("unreachable line...")
        generated_sentences.append(generated_sentence)

    return generated_sentences


def _shuffle_within_segments(tags, replace_ratio):
    '''
    Given a segmented sentence such as ["O", "O", "B-PER", "I-PER", "I-PER", "B-ORG", "B-ORG", "I-ORG", "I-ORG"],
    shuffle the token order within each segment
    '''
    segments = [0]
    for i, tag in enumerate(tags):
        if i == 0: continue
        if tag == "O":
            if tags[i - 1] == "O":
                segments.append(segments[-1])
            else:
                segments.append(segments[-1] + 1)
        elif tag.startswith("B"):
            segments.append(segments[-1] + 1)
        else:
            segments.append(segments[-1])

    # segments: [0 0 1 1 1 2 3 3 3]

    shuffled_idx = []
    start, end = 0, 0
    while start < len(segments) and end < len(segments):
        while end < len(segments) and segments[end] == segments[start]:
            end += 1
        segment = [i for i in range(start, end)]
        if len(segment) > 1 and np.random.binomial(1, replace_ratio, 1)[0] == 1:
            random.shuffle(segment)
        shuffled_idx += segment
        start = end
    return shuffled_idx


def generate_sentences_by_shuffle_within_segments(sentence, replace_ratio, num_generated_samples):
    sentences = []
    for i in range(num_generated_samples):
        generated_sentence = Sentence("%s-shuffle-within--segments-%d" % (sentence.idx, i))
        tags = [token.get_label("gold") for token in sentence.tokens]
        shuffled_idx = _shuffle_within_segments(tags, replace_ratio)
        assert len(shuffled_idx) == len(tags)
        for i, tag in zip(shuffled_idx, tags):
            generated_token = Token(sentence[i].text)
            generated_token.set_label("gold", tag)
            generated_sentence.add_token(generated_token)
        sentences.append(generated_sentence)
    return sentences


def get_label2tokens(dataset, p_power):
    token_freq = {}
    for sentence in dataset:
        for token in sentence:
            if token.text.lower() in STOPWORDS: continue
            label = token.get_label("gold")
            if label not in token_freq: token_freq[label] = defaultdict(int)
            token_freq[label][token.text] += 1

    label2tokens = {}
    for label in token_freq:
        tokens, values = [], []
        for t in token_freq[label]:
            tokens.append(t)
            values.append(np.power(token_freq[label][t], p_power))
        total_values = sum(values)
        probabilities = [v / total_values for v in values]
        label2tokens[label] = (tokens, probabilities)

    return label2tokens


def generate_sentences_by_replace_token(sentence, label2tokens, replace_ratio, num_generated_samples):
    sentences = []
    for i in range(num_generated_samples):
        generated_sentence = Sentence("%s-replace-token-%d" % (sentence.idx, i))
        masks = np.random.binomial(1, replace_ratio, len(sentence))
        for mask, token in zip(masks, sentence.tokens):
            label = token.get_label("gold")
            if mask == 0 or token.text.lower() in STOPWORDS:
                generated_token = Token(token.text)
            else:
                random_idx = np.random.choice(len(label2tokens[label][1]), 1, p=label2tokens[label][1])[0]
                generated_token = Token(label2tokens[label][0][random_idx])
            generated_token.set_label("gold", label)
            generated_sentence.add_token(generated_token)
        sentences.append(generated_sentence)
    return sentences


def generate_sentences_by_synonym_replacement(sentence, replace_ratio, num_generated_samples):
    sentences = []
    for i in range(num_generated_samples):
        generated_sentence = Sentence("%s-synonym-replacement-%d" % (sentence.idx, i))
        masks = np.random.binomial(1, replace_ratio, len(sentence))
        for mask, token in zip(masks, sentence.tokens):
            label = token.get_label("gold")
            if mask == 0 or token.text.lower() in STOPWORDS:
                generated_token = Token(token.text)
                generated_token.set_label("gold", label)
                generated_sentence.add_token(generated_token)
            else:
                synonyms = set()
                for syn in wordnet.synsets(token.text):
                    for l in syn.lemmas():
                        synonym = l.name().replace("_", " ").replace("-", " ")
                        synonyms.add(synonym)
                if token.text in synonyms:
                    synonyms.remove(token.text)
                if len(synonyms) == 0:
                    generated_token = Token(token.text)
                    generated_token.set_label("gold", label)
                    generated_sentence.add_token(generated_token)
                    continue

                synonym = random.choice(list(synonyms))
                for s_i, s_token in enumerate(synonym.split()):
                    generated_token = Token(s_token)
                    if s_i > 0 and label.startswith("B-"): label = "I-%s" % label[2:]
                    generated_token.set_label("gold", label)
                    generated_sentence.add_token(generated_token)
        sentences.append(generated_sentence)
    return sentences