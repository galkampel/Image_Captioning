
import argparse
import os
import torch
import torch.optim as optim
import spacy
from torchvision.transforms import transforms
from torch.utils.tensorboard import SummaryWriter
from utils.ic_dataloader import *
from preprocesssing import *


def get_arguments(arg_list=None):
    parser = argparse.ArgumentParser(description="Model parameters")
    ###### dataset parameters ######
    parser.add_argument("--folder_params", type=str, default="input", choices=["input"])
    parser.add_argument("--file_params", type=str, default="pubmed_gatgcn_hyper_params", choices=[
        "pubmed_gatgat_params", "pubmed_gatgat_hyper_params", "pubmed_gatgcn_params",
        "pubmed_gatgcn_hyper_params", "pubmed_gcngcn_params", "pubmed_gcngcn_hyper_params",
        "cora_gatgat_params", "cora_gatgat_hyper_params", "cora_gatgcn_params",
        "cora_gatgcn_hyper_params", "cora_gcngcn_params", "cora_gcngcn_hyper_params",
    ])
    return parser.parse_args(arg_list)

# download: python -m spacy download en (en_core_web_sm) for english spacy tokenization
def main(args):

    parser = spacy.load('en') if args.word_preprocessing else None
    transform = None
    vocab = Vocabulary(parser)
    root = os.path.join(os.getcwd(), args.root)
    caption_file = os.path.join(root, args.caption_file)
    dataset = CaptionDataset(root, caption_file, vocab, transform)
    tr_te_lengths = [sum(args.lengths[:-1]) , args.lengths[-1]]
    train_val_data, test_data = train_test_split(dataset, tr_te_lengths)
    train_data, val_data = train_test_split(train_val_data, args.lengths[:-1])

    train_loader = get_loader(train_data, use_collate=args.use_collate)
    val_loader = get_loader(val_data, use_collate=args.use_collate)
    test_loader = get_loader(test_data, use_collate=args.use_collate)


if __name__ == "__main__":
    arguments = get_arguments()
    main(arguments)