"""
Code for simple data augmentation methods for named entity recognition (Coling 2020).
Copyright (c) 2020 - for information on the respective copyright owner see the NOTICE file.

SPDX-License-Identifier: Apache-2.0

The code in this file is partly based on the FLAIR library,
(https://github.com/flairNLP/flair), licensed under the MIT license,
cf. 3rd-party-licenses.txt file in the root directory of this source tree.
"""

import itertools, logging, os, time, torch
from collections import defaultdict
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data import get_spans
from augment import generate_sentences_by_shuffle_within_segments, generate_sentences_by_replace_mention, generate_sentences_by_replace_token, generate_sentences_by_synonym_replacement


logger = logging.getLogger(__name__)


class Metric:
    def __init__(self):
        self._tps = defaultdict(int)
        self._fps = defaultdict(int)
        self._tns = defaultdict(int)
        self._fns = defaultdict(int)

    def add_tp(self, class_name):
        self._tps[class_name] += 1

    def add_fp(self, class_name):
        self._fps[class_name] += 1

    def add_tn(self, class_name):
        self._tns[class_name] += 1

    def add_fn(self, class_name):
        self._fns[class_name] += 1

    def get_tp(self, class_name=None):
        return sum(self._tps.values()) if class_name is None else self._tps[class_name]

    def get_fp(self, class_name=None):
        return sum(self._fps.values()) if class_name is None else self._fps[class_name]

    def get_tn(self, class_name=None):
        return sum(self._tns.values()) if class_name is None else self._tns[class_name]

    def get_fn(self, class_name=None):
        return sum(self._fns.values()) if class_name is None else self._fns[class_name]

    def f_score(self, class_name=None):
        tp = self.get_tp(class_name)
        fp = self.get_fp(class_name)
        fn = self.get_fn(class_name)
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        return precision, recall, f1

    def accuracy(self, class_name=None):
        tp = self.get_tp(class_name)
        fp = self.get_fp(class_name)
        fn = self.get_fn(class_name)
        return tp / (tp + fp + fn) if tp + fp + fn > 0 else 0.0

    def micro_avg_f_score(self):
        return self.f_score()[-1]

    def macro_avg_f_score(self):
        scores = [self.f_score(c)[-1] for c in self.get_classes()]
        return sum(scores) / len(scores) if len(scores) > 0 else 0.0

    def micro_avg_accuracy(self):
        return self.accuracy()

    def macro_avg_accuracy(self):
        accuracies = [self.accuracy(c) for c in self.get_classes()]
        return sum(accuracies) / len(accuracies) if len(accuracies) > 0 else 0.0

    def get_classes(self):
        all_classes = set(list(self._tps.keys()) + list(self._fps.keys()) + list(self._tns.keys()) + list(self._fns.keys()))
        return sorted([c for c in all_classes if c is not None])

    def to_dict(self):
        result = {}
        for n in self.get_classes():
            result[n] = {"tp": self.get_tp(n), "fp": self.get_fp(n), "fn": self.get_fn(n), "tn": self.get_tn(n)}
            result[n]["p"], result[n]["r"], result[n]["f"] = self.f_score(n)
        result["overall"] = {"tp": self.get_tp(), "fp": self.get_fp(), "fn": self.get_fn(), "tn": self.get_tn()}
        result["overall"]["p"], result["overall"]["r"], result["overall"]["f"] = self.f_score()
        return result


def evaluate(encoder, mlp, crf, data_loader, output_filepath=None, verbose=False):
    with torch.no_grad():
        start_time = time.time()
        for sentences in data_loader:
            preds = crf.viterbi_tags(mlp(encoder.forward(sentences)), sentences)
            for s, pred in zip(sentences, preds):
                for t, p in zip(s, pred):
                    t.set_label("pred", p)
                s.clear_embeddings()
        logger.info(f"Finish evaluation: {time.time() - start_time} s")

    metric = Metric()
    for sentences in data_loader:
        for s in sentences:
            gold_spans = [(span.label, str(span)) for span in get_spans(s, "gold")]
            pred_spans = [(span.label, str(span)) for span in get_spans(s, "pred")]
            for pred_span in pred_spans:
                if pred_span in gold_spans:
                    metric.add_tp(pred_span[0])
                else:
                    metric.add_fp(pred_span[0])
            for gold_span in gold_spans:
                if gold_span not in pred_spans:
                    metric.add_fn(gold_span[0])
                else:
                    metric.add_tn(gold_span[0])
    logger.info(f"micro-avg: acc {metric.micro_avg_accuracy()} - micro-avg-f1-score {metric.micro_avg_f_score()}")

    if verbose:
        logger.info("\t".join(["Class", "TP", "TP", "FN", "TN", "Precision", "Recall", "F1"]))
        for n in metric.get_classes():
            tp, fp, fn, tn = metric.get_tp(n), metric.get_fp(n), metric.get_fn(n), metric.get_tn(n)
            p, r, f = metric.f_score(n)
            logger.info("%s\t%d\t%d\t%d\t%d\t%.4f\t%.4f\t%.4f" % (n, tp, fp, fn, tn, p, r, f))

    if output_filepath is not None:
        with open(output_filepath, "w", encoding="utf-8") as f:
            for sentences in data_loader:
                for s in sentences:
                    for t in s:
                        f.write("%s %s %s\n" % (t.text, t.get_label("gold"), t.get_label("pred")))
                    f.write("\n")
    return metric


def final_test(args, encoder, mlp, crf, test_data, name):
    logger.info("-" * 100)
    logger.info("Testing using best model ...")
    encoder.eval()
    mlp.eval()
    crf.eval()
    encoder.load_state_dict(torch.load(os.path.join(args.output_dir, "encoder.pt")))
    mlp.load_state_dict(torch.load(os.path.join(args.output_dir, "mlp.pt")))
    crf.load_state_dict(torch.load(os.path.join(args.output_dir, "crf.pt")))
    data_loader = DataLoader(test_data, batch_size=args.eval_bs, collate_fn=list)
    test_score = evaluate(encoder, mlp, crf, data_loader, os.path.join(args.output_dir, "%s.tsv" % name), True)
    return test_score.to_dict()


def train_epoch(args, encoder, mlp, crf, optimizer, train_data, epoch, category2mentions, label2tokens):
    if len(args.augmentation) > 0:
        augmented_sentences = []
        for s in train_data:
            if "MR" in args.augmentation:
                augmented_sentences += generate_sentences_by_replace_mention(s, category2mentions, args.replace_ratio,
                                                                             args.num_generated_samples)
            if "LwTR" in args.augmentation:
                augmented_sentences += generate_sentences_by_replace_token(s, label2tokens, args.replace_ratio,
                                                                           args.num_generated_samples)
            if "SiS" in args.augmentation:
                augmented_sentences += generate_sentences_by_shuffle_within_segments(s, args.replace_ratio,
                                                                                     args.num_generated_samples)
            if "SR" in args.augmentation:
                augmented_sentences += generate_sentences_by_synonym_replacement(s, args.replace_ratio,
                                                                                 args.num_generated_samples)
        train_data += augmented_sentences
    else:
        logger.info("No data augmentation used")

    logger.info("-" * 100)
    logger.info(f"# sentences and augmented sentences: {len(train_data)}")
    data_loaders = DataLoader(train_data, args.train_bs, shuffle=True, collate_fn=list)
    iterator = iter(data_loaders)
    total_loss, num_batches = 0, len(data_loaders)
    logging_intervals, epoch_state_time = max(1, int(num_batches / 10)), time.time()
    for i in range(num_batches):
        encoder.zero_grad()
        mlp.zero_grad()
        crf.zero_grad()
        sentences = next(iterator)
        features = mlp(encoder.forward(sentences))
        loss = crf.forward_loss(features, sentences)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5.0)
        torch.nn.utils.clip_grad_norm_(mlp.parameters(), 5.0)
        torch.nn.utils.clip_grad_norm_(crf.parameters(), 5.0)
        optimizer.step()
        total_loss += loss.item()
        if i % logging_intervals == 0:
            logging_loss = total_loss / (i + 1)
            logging_speed = args.train_bs * (i + 1) / (time.time() - epoch_state_time)
            logger.info(f"epoch {epoch}/{args.max_epochs} - batch {i + 1}/{num_batches} - "
                        f"loss {logging_loss} - samples/second: {logging_speed}")

    for s in train_data:
        s.clear_embeddings()

    return total_loss / len(train_data)


def _evaluate_after_epoch(args, encoder, mlp, crf, eval_data, scheduler, optimizer, prev_lr):
    data_loader = DataLoader(eval_data, args.eval_bs, collate_fn=list)
    score = evaluate(encoder, mlp, crf, data_loader, verbose=args.debug).micro_avg_f_score()
    scheduler.step(score)
    for group in optimizer.param_groups:
        lr = group["lr"]
    if lr != prev_lr: logger.info(f"change lr from {prev_lr} to {lr}")
    return score, lr


def train(args, encoder, mlp, crf, train_data, dev_data, category2mentions, label2tokens):
    logger.info(f"# sentences in training set: {len(train_data)}")
    logger.info(f"# sentences in development set: {len(dev_data)}")

    assert args.optimizer.lower() in ["sgd", "adam"], "Unknown optimizer"
    optimizer = torch.optim.SGD if args.optimizer.lower() == "sgd" else torch.optim.AdamW
    parameters = [encoder.parameters()] + [mlp.parameters()] + [crf.parameters()]
    optimizer = optimizer(itertools.chain(*map(list, parameters)), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, factor=args.anneal_factor, patience=args.anneal_patience, mode="max")

    for epoch in range(1, args.max_epochs + 1):
        encoder.train()
        mlp.train()
        crf.train()
        for group in optimizer.param_groups:
            lr = group["lr"]
        if lr < args.min_lr:
            logger.info("learning rate too small -- quitting training!")
            break

        epoch_start_time = time.time()
        train_loss = train_epoch(args, encoder, mlp, crf, optimizer, train_data, epoch, category2mentions, label2tokens)
        encoder.eval()
        mlp.eval()
        crf.eval()
        dev_score, lr = _evaluate_after_epoch(args, encoder, mlp, crf, dev_data, scheduler, optimizer, lr)
        args.result["epoch-%d" % epoch] = {"time": time.time() - epoch_start_time,
                                           "lr": lr, "train_loss": train_loss, "dev_score": dev_score}

        if dev_score == scheduler.best or "best_epoch" not in args.result:
            args.result["best_epoch"] = epoch
            logger.info("New best model found")
            torch.save(encoder.state_dict(), os.path.join(args.output_dir, "encoder.pt"))
            torch.save(mlp.state_dict(), os.path.join(args.output_dir, "mlp.pt"))
            torch.save(crf.state_dict(), os.path.join(args.output_dir, "crf.pt"))
        else:
            logger.info(f"No improvement since last {epoch - args.result['best_epoch']} epochs, "
                        f"best score is {scheduler.best}")
            if epoch - args.result["best_epoch"] >= args.early_stop_patience:
                logger.info(f"Early stop since no improvement since last {args.early_stop_patience} epochs")
                break