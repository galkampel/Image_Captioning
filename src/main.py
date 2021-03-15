import argparse
import spacy
from src.dataloader import *
from dataset import *
from src.train import Trainer
from torchvision.transforms import transforms
import torch.nn as nn
import torch.optim as optim


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


# TODO:
#    i.  create a flow for training
#   ii. create a different optimizer for encoder (if needed) and decoder (done)
#  iii. put self.encoder.unfreeze_model_weights(epoch) outside (in the epochs loop)
#   iv. print training loss (overfit the data)


def train(trainer, dataloader):
    pass


def predict():
    pass


def visualize():
    pass


# download: python -m spacy download en (en_core_web_sm) for english spacy tokenization
def main(args):

    parser = spacy.load('en') if args.word_preprocessing else None
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    vocab = Vocabulary(parser)  # freq_thres=2 (default)
    root = os.path.join(os.getcwd(), args.root)  # root = 'flickr8k'
    caption_file = os.path.join(root, args.caption_file)  # caption_file='captions.txt'
    img_folder = os.path.join(root, args.img_folder)  # img_folder = 'images'
    dataset = CaptionDataset(img_folder, caption_file, vocab, transform)
    training_set, validation_set, test_set = train_test_split(dataset, args.ratios)  # ratios=fractions for split (list)
    # dataloader_params: batch_size: [32, 64], use_collate=True,  n_workers: [8, 2], pin_memory = True
    train_loader = get_loader(training_set, shuffle=True, ** args.dataloader_params)
    val_loader = get_loader(validation_set, ** args.dataloader_params)
    test_loader = get_loader(test_set, ** args.dataloader_params)

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<PAD>"])
    # cuda_id- the cuda to run
    device = f'cuda:{args.cuda_id}' if torch.cuda.is_available() else 'cpu'
    vocab_size = len(vocab)
    enc_dim = args.enc_dim
    dec_dim = args.dec_dim

    optimizer_dict = {'adam': optim.Adam, 'sgd': optim.SGD}
    optimizer_dec = optim.Adam
    optimizer_enc = ...

    trainer = Trainer(criterion, device, vocab_size, enc_dim, dec_dim, ** args.model_params)
    #  trainer.set_optimizer()


if __name__ == "__main__":
    arguments = get_arguments()
    main(arguments)
