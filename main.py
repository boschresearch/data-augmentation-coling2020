"""
Code for simple data augmentation methods for named entity recognition (Coling 2020).
Copyright (c) 2020 - for information on the respective copyright owner see the NOTICE file.

SPDX-License-Identifier: Apache-2.0
"""

import argparse, json, logging, numpy, os, random, sys, torch

from data import ConllCorpus
from train import train, final_test
from models import TransformerEncoder, LinearCRF, MLP
from augment import get_category2mentions, get_label2tokens


logger = logging.getLogger(__name__)


def parse_parameters(parser=None):
    if parser is None: parser = argparse.ArgumentParser()

    # input
    parser.add_argument("--data_folder", default="/data/dai031/Corpora")
    parser.add_argument("--task_name", default="development", type=str)
    parser.add_argument("--train_filepath", default="train.txt", type=str)
    parser.add_argument("--dev_filepath", default="dev.txt", type=str)
    parser.add_argument("--test_filepath", default="test.txt", type=str)

    # output
    parser.add_argument("--output_dir", default="development", type=str)
    parser.add_argument("--result_filepath", default="development.json", type=str)
    parser.add_argument("--log_filepath", default="development.log")

    # train
    parser.add_argument("--lr", default=3e-5, type=float)
    parser.add_argument("--min_lr", default=1e-8, type=float)
    parser.add_argument("--train_bs", default=16, type=int)
    parser.add_argument("--eval_bs", default=16, type=int)
    parser.add_argument("--max_epochs", default=100, type=int)
    parser.add_argument("--anneal_factor", default=0.5, type=float)
    parser.add_argument("--anneal_patience", default=3, type=int)
    parser.add_argument("--early_stop_patience", default=10, type=int)
    parser.add_argument("--optimizer", default="adam", type=str)

    # environment
    parser.add_argument("--seed", default=52, type=int)
    parser.add_argument("--device", default=0, type=int)

    # embeddings
    parser.add_argument("--embedding_type", default=None, type=str)
    parser.add_argument("--pretrained_dir", default=None, type=str)

    # dropout
    parser.add_argument("--dropout", default=0.4, type=float)
    parser.add_argument("--word_dropout", default=0.05, type=float)
    parser.add_argument("--variational_dropout", default=0.5, type=float)

    # augmentation
    parser.add_argument("--augmentation", type=str, nargs="+", default=[])
    parser.add_argument("--p_power", default=1.0, type=float,
                        help="the exponent in p^x, used to smooth the distribution, "
                             "if it is 1, the original distribution is used; "
                             "if it is 0, it becomes uniform distribution")
    parser.add_argument("--replace_ratio", default=0.3, type=float)
    parser.add_argument("--num_generated_samples", default=1, type=int)

    parser.add_argument("--debug", action="store_true")

    args, _ = parser.parse_known_args()

    args.train_filepath = os.path.join(args.data_folder, args.train_filepath)
    args.dev_filepath = os.path.join(args.data_folder, args.dev_filepath)
    args.test_filepath = os.path.join(args.data_folder, args.test_filepath)

    return args


def random_seed(seed=52):
    if seed > 0:
        random.seed(seed)
        numpy.random.seed(int(seed / 2))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(int(seed / 4))
        torch.cuda.manual_seed(int(seed / 8))
        torch.cuda.manual_seed_all(int(seed / 8))


if __name__ == "__main__":
    args = parse_parameters()
    device = torch.device("cuda:%d" % args.device)
    args.result = {}

    handlers = [logging.FileHandler(filename=args.log_filepath), logging.StreamHandler(sys.stdout)]
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s", datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO, handlers=handlers)

    if not os.path.exists(args.output_dir): os.makedirs(args.output_dir, exist_ok=True)
    random_seed(args.seed)
    logger.info(f'CONFIG: "{args}"')

    corpus = ConllCorpus(args.task_name, args.train_filepath, args.dev_filepath, args.test_filepath)
    tag_dict = corpus.build_tag_dict("gold")

    category2mentions = get_category2mentions(corpus.train)
    label2tokens = get_label2tokens(corpus.train, args.p_power)

    encoder = TransformerEncoder(args, device)
    mlp = MLP(encoder.output_dim, len(tag_dict), encoder.output_dim, 1).to(device)
    crf = LinearCRF(tag_dict, device)
    dev_scores = train(args, encoder, mlp, crf, corpus.train, corpus.dev, category2mentions, label2tokens)

    args.result["dev_result"] = final_test(args, encoder, mlp, crf, corpus.dev, "dev")
    args.result["test_result"] = final_test(args, encoder, mlp, crf, corpus.test, "test")

    with open(args.result_filepath, "w") as f:
        json.dump(vars(args), f, indent=4)