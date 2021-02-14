
import argparse
import os
from utils.ic_dataloader import *



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


def main(args):
    dataset =  ...
    tr_te_lengths = [sum(args.lengths[:-1]) , args.lengths[-1]]
    train_val_data, test_data = train_test_split(dataset, tr_te_lengths)
    train_data, val_data = train_test_split(train_val_data, args.lengths[:-1])
    train_loader = get_loader(dataset, )




if __name__ == "__main__":
    arguments = get_arguments()
    main(arguments)