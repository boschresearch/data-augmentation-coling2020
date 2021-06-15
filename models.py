"""
Code for simple data augmentation methods for named entity recognition (Coling 2020).
Copyright (c) 2020 - for information on the respective copyright owner see the NOTICE file.

SPDX-License-Identifier: Apache-2.0

The code in this file is partly based on the FLAIR library,
(https://github.com/flairNLP/flair), licensed under the MIT license,
and on the AllenNLP library (https://github.com/allenai/allennlp-models),
licensed under the Apache 2.0 license,
cf. 3rd-party-licenses.txt file in the root directory of this source tree.
"""

import logging, torch
from typing import List
from transformers import BertTokenizer, BertConfig, BertModel
from data import Sentence


logger = logging.getLogger(__name__)


def pad_tensors(tensor_list: List, lengths):
    '''assume the inputs have shape like (batch_size, length, ...)'''
    shape = [len(tensor_list), max(lengths)] + list(tensor_list[0].shape[1:])
    padded_tensors = torch.zeros(*shape, dtype=torch.long)
    for i, tensor in enumerate(tensor_list):
        padded_tensors[i, : lengths[i]] = tensor
    return padded_tensors


def get_mask_from_lengths(lengths: List[int]):
    range_tensor = torch.ones((len(lengths), max(lengths))).cumsum(dim=1)
    return torch.tensor(lengths).unsqueeze(1) >= range_tensor


class MLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=0, n_layers=0, name="mlp"):
        super(MLP, self).__init__()
        self.nn = torch.nn.Sequential()
        for i in range(n_layers):
            self.nn.add_module(f"{name}-hidden-{i}", torch.nn.Linear(input_dim if i == 0 else hidden_size, hidden_size))
        self.nn.add_module(f"{name}-output", torch.nn.Linear(hidden_size if n_layers > 0 else input_dim, output_dim))

    def forward(self, inputs):
        return self.nn(inputs)


class TransformerEncoder(torch.nn.Module):
    def __init__(self, args, devive):
        super(TransformerEncoder, self).__init__()
        assert args.embedding_type == "bert"
        self.embedding_type = args.embedding_type
        self.device = devive
        self.tokenizer = BertTokenizer.from_pretrained(args.pretrained_dir)
        self.config = BertConfig.from_pretrained(args.pretrained_dir, output_hidden_states=True)
        self.model = BertModel.from_pretrained(pretrained_model_name_or_path=args.pretrained_dir, config=self.config)
        self.model.eval()
        self.dropout = torch.nn.Dropout(args.dropout) if args.dropout > 0.0 else None
        self.word_dropout = WordDropout(args.word_dropout) if args.word_dropout > 0.0 else None
        self.output_dim = self.config.hidden_size

        self.to(self.device)

    def convert_token_to_subtokens(self, sentence):
        '''Mainly count the number of subtokens for each token'''
        tokens = []
        for t in sentence:
            subtokens = self.tokenizer.tokenize(t.text)
            tokens.append(t.text if subtokens else self.tokenizer.unk_token)
            t.num_subtokens = len(subtokens) if subtokens else 1
        return " ".join(tokens)

    def _get_transformer_embeddings(self, sentences, fine_tune=False, bos_token="[CLS]", eos_token="[SEP]"):
        gradient_context = torch.enable_grad() if fine_tune else torch.no_grad()
        with gradient_context:
            inputs, lengths = [], []
            for s in sentences:
                if s.tokens_indices is None:
                    subtokens = self.tokenizer.tokenize(self.convert_token_to_subtokens(s))
                    s.tokens_indices = self.tokenizer.convert_tokens_to_ids([bos_token] + subtokens + [eos_token])
                inputs.append(s.tokens_indices)
                lengths.append(len(s.tokens_indices))

            inputs = [v + [self.tokenizer.pad_token_id] * (max(lengths) - len(v)) for v in inputs]
            inputs = torch.tensor(inputs).to(self.device)
            masks = get_mask_from_lengths(lengths).to(self.device)
            hidden_states = self.model(inputs, masks)[0]

            for outputs, sentence in zip(hidden_states, sentences):
                offset = 1
                for token in sentence:
                    token.set_embedding(self.embedding_type, torch.cat([outputs[offset]]), self.device)
                    offset += token.num_subtokens

    def forward(self, sentences: List[Sentence]):
        if type(sentences) is Sentence: sentences = [sentences]
        self._get_transformer_embeddings(sentences, fine_tune=self.training)
        max_len = max([len(s) for s in sentences])
        zero_tensor = torch.zeros(self.output_dim * max_len, dtype=torch.float, device=self.device)

        outputs = []
        for s in sentences:
            outputs += [e for t in s for e in t.get_embedding_list(self.device)]
            padding_length = max_len - len(s)
            if padding_length > 0:
                outputs.append(zero_tensor[:self.output_dim * padding_length])

        outputs = torch.cat(outputs).view([len(sentences), max_len, self.output_dim])
        if self.dropout is not None: outputs = self.dropout(outputs)
        if self.word_dropout is not None: outputs = self.word_dropout(outputs)
        return outputs


def viterbi_decode(tag_sequence, transition_matrix):
    sequence_length, num_tags = tag_sequence.size()
    path_scores, path_indices = [tag_sequence[0, :]], []
    for timestep in range(1, sequence_length):
        summed_potentials = path_scores[timestep - 1].unsqueeze(-1) + transition_matrix # (num_tags, num_tags)
        scores, paths = torch.max(summed_potentials, 0)
        path_scores.append(tag_sequence[timestep, :] + scores.squeeze())
        path_indices.append(paths.squeeze())
    viterbi_score, best_path = torch.max(path_scores[-1], 0)
    viterbi_path = [int(best_path.numpy())]
    for backward_timestep in reversed(path_indices):
        viterbi_path.append(int(backward_timestep[viterbi_path[-1]]))
    viterbi_path.reverse()
    return viterbi_path, viterbi_score


def _is_transition_allowed(constraint_type, from_prefix, from_entity, to_prefix, to_entity):
    assert constraint_type == "BIO"
    if to_prefix == "START" or from_prefix == "END": return False
    if from_prefix == "START": return to_prefix in ("O", "B")
    if to_prefix in ("END", "O", "B"): return True
    if to_prefix == "I" and from_prefix in ("B", "I") and from_entity == to_entity: return True
    return False


def allowed_transitions(constraint_type, tag2idx):
    assert constraint_type == "BIO"
    tags_with_boundaries = list(tag2idx.items()) + [("START", len(tag2idx)), ("END", len(tag2idx) + 1)]
    allowed = []
    for from_tag, from_tag_idx in tags_with_boundaries:
        if from_tag in ("START", "END", "O"):
            from_prefix, from_entity = from_tag, ""
        else:
            from_prefix, from_entity = from_tag.split("-")
        for to_tag, to_tag_idx in tags_with_boundaries:
            if to_tag in ("START", "END", "O"):
                to_prefix, to_entity = to_tag, ""
            else:
                to_prefix, to_entity = to_tag.split("-")
            if _is_transition_allowed(constraint_type, from_prefix, from_entity, to_prefix, to_entity):
                allowed.append((from_tag_idx, to_tag_idx))
    return allowed


class LinearCRF(torch.nn.Module):
    def __init__(self, tag_dict, device):
        super().__init__()
        self.tag_dict = tag_dict
        self.device = device
        self.trans = torch.nn.Parameter(torch.Tensor(len(tag_dict), len(tag_dict)))

        constrained = torch.Tensor(len(tag_dict) + 2, len(tag_dict) + 2).fill_(0.0)
        for i, j in allowed_transitions("BIO", tag_dict.item2idx):
            constrained[i, j] = 1.0
        self.constrained = torch.nn.Parameter(constrained, requires_grad=False)
        self.starts = torch.nn.Parameter(torch.Tensor(len(tag_dict)))
        self.ends = torch.nn.Parameter(torch.Tensor(len(tag_dict)))

        torch.nn.init.xavier_normal_(self.trans)
        torch.nn.init.normal_(self.starts)
        torch.nn.init.normal_(self.ends)

        self.to(self.device)

    def forward_loss(self, features, sentences):
        lengths = [len(s) for s in sentences]
        gold_tags = [torch.tensor([self.tag_dict.get_idx(t.get_label("gold")) for t in s]) for s in sentences]
        gold_tags = pad_tensors(gold_tags, lengths).to(self.device)
        mask = get_mask_from_lengths(lengths).to(self.device)
        return torch.mean(self._input_likelihood(features, mask) - self._joint_likelihood(features, gold_tags, mask))

    def _joint_likelihood(self, logits, tags, mask):
        batch_size, seq_len, _ = logits.size()
        logits = logits.transpose(0, 1).contiguous()
        tags = tags.transpose(0, 1).contiguous()
        mask = mask.transpose(0, 1).contiguous()
        score = self.starts.index_select(0, tags[0]) # (batch_size)
        for i in range(seq_len - 1):
            current_tag, next_tag = tags[i], tags[i + 1] # (batch_size)
            transition_score = self.trans[current_tag.view(-1), next_tag.view(-1)] # (batch_size)
            emit_score = logits[i].gather(1, current_tag.view(batch_size, 1)).squeeze(1) # (batch_size)
            score += transition_score * mask[i + 1] + emit_score * mask[i]
        last_tag_idx = mask.sum(0).long() - 1
        last_tags = tags.gather(0, last_tag_idx.view(1, batch_size)).squeeze(0)
        last_inputs = logits[-1] # (batch_size, num_tags)
        last_input_score = last_inputs.gather(1, last_tags.view(-1, 1)).squeeze() # (batch_size, )
        score += last_input_score * mask[-1]
        score += self.ends.index_select(0, last_tags)
        return score

    def _input_likelihood(self, logits, mask):
        batch_size, seq_len, num_tags = logits.size()
        logits = logits.transpose(0, 1).contiguous()
        mask = mask.transpose(0, 1).contiguous()

        alpha = self.starts.view(1, num_tags) + logits[0] # (batch_size, num_tags)
        for i in range(1, seq_len):
            emit_scores = logits[i].view(batch_size, 1, num_tags)
            transition_scores = self.trans.view(1, num_tags, num_tags)
            broadcast_alpha = alpha.view(batch_size, num_tags, 1)
            inner = broadcast_alpha + emit_scores + transition_scores # (batch_size, num_tags, num_tags)
            alpha *= (~mask[i]).view(batch_size, 1)
            alpha += torch.logsumexp(inner, dim=1) * mask[i].view(batch_size, 1)
        stops = alpha + self.ends.view(1, num_tags)
        return torch.logsumexp(stops, dim=-1)

    def viterbi_tags(self, logits, sentences):
        lengths = [len(s) for s in sentences]
        mask = get_mask_from_lengths(lengths)
        _, seq_len, num_tags = logits.size()
        logits, mask = logits.data, mask.data
        start_tag, end_tag = num_tags, num_tags + 1
        trans = torch.Tensor(num_tags + 2, num_tags + 2).fill_(-10000.0)
        constrained = self.trans * self.constrained[:num_tags, :num_tags] \
                      + -10000.0 * (1 - self.constrained[:num_tags, :num_tags])
        trans[:num_tags, :num_tags] = constrained.data
        trans[start_tag, :num_tags] = self.starts.detach() * self.constrained[start_tag, :num_tags].data \
                                      + -10000.0 * (1 - self.constrained[start_tag, :num_tags].detach())
        trans[:num_tags, end_tag] = self.ends.detach() * self.constrained[:num_tags, end_tag].data \
                                    + -10000.0 * (1 - self.constrained[:num_tags, end_tag].detach())
        preds = []
        tag_sequence = torch.Tensor(seq_len + 2, num_tags + 2)
        for i, logit in enumerate(logits):
            tag_sequence.fill_(-10000.0)
            tag_sequence[0, start_tag] = 0.0
            tag_sequence[1: (lengths[i] + 1), :num_tags] = logit[:lengths[i]]
            tag_sequence[lengths[i] + 1, end_tag] = 0.0
            pred_path, _ = viterbi_decode(tag_sequence[: (lengths[i] + 2)], trans)
            preds.append([self.tag_dict.get_item(p) for p in pred_path[1:-1]])
        return preds


class WordDropout(torch.nn.Module):
    '''Randomly drop out the entire word or character'''
    def __init__(self, dropout=0.05):
        super(WordDropout, self).__init__()
        self.dropout = dropout

    def forward(self, inputs):
        if not self.training or not self.dropout: return inputs
        batch_size, seq_len, _ = inputs.size()
        mask = inputs.data.new(batch_size, seq_len, 1).bernoulli_(1 - self.dropout)
        mask = torch.autograd.Variable(mask, requires_grad=False)
        return mask * inputs