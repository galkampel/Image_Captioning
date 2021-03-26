import os
import pandas as pd
import json
import torch
from torch.utils.data import Dataset, random_split
from PIL import Image


def train_test_split(dataset, ratios):
    N = len(dataset)
    lengths = [int(N * ratio) for ratio in ratios]
    resid = N - sum(lengths)
    lengths[0] += resid
    return random_split(dataset, lengths)


class Vocabulary:
    def __init__(self, parser, freq_thres=2):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}  # unk - oov or less than freq_thres
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.parser = parser  # e.g. spacy_eng
        self.freq_thres = freq_thres

    def __len__(self):
        return len(self.itos)

    def tokenize(self, text):
        return [tok.text.lower() for tok in self.parser(text)]

    def build_vocab(self, sentences):  # update itos and stoi
        freqs = {}
        idx = 4
        for sentence in sentences:
            for word in self.tokenize(sentence):
                if word not in freqs:
                    freqs[word] = 1
                else:
                    freqs[word] += 1
                if freqs[word] == self.freq_thres:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenize(text)
        return [self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
                for token in tokenized_text]

    def load_vocab_dicts(self, folder_name, filename):
        stoi_folder = os.path.join(os.getcwd(), folder_name)
        os.makedirs(stoi_folder, exist_ok=True)
        full_path = os.path.join(stoi_folder, f'{filename}.json')
        with open(full_path) as json_file:
            params = json.load(json_file)
        self.stoi = params["stoi"]
        self.itos = params["itos"]


# each image has ~5 corresponding images
class CaptionDataset(Dataset):
    def __init__(self, image_folder, load_folder, vocab, transform=None):
        self.image_folder = image_folder
        self.load_folder = os.path.join(os.getcwd(), load_folder)
        self.vocab = vocab
        self.load_vocab()
        self.transform = transform
        self.df = self.load_df()
        self.imgs = self.df['image']
        self.captions = self.df['caption']
        self.idx2captions = self.load_idx2captions()

    def load_df(self, filename='df'):
        full_path = os.path.join(self.load_folder, f'{filename}.csv')
        return pd.read_csv(full_path)

    def load_idx2captions(self, file_name='idx2captions'):
        full_path = os.path.join(self.load_folder, f'{file_name}.json')
        with open(full_path) as json_file:
            params = json.load(json_file)
        return {int(key): val for key, val in params['idx2captions'].items()}

    def load_vocab(self, file_name='vocab'):
        full_path = os.path.join(self.load_folder, f'{file_name}.json')
        with open(full_path) as json_file:
            params = json.load(json_file)
        self.vocab.stoi = params['stoi']
        self.vocab.itos = {int(idx):token for idx, token in params['itos'].items()}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        caption = self.captions[idx]
        img_id = self.imgs[idx]
        img = Image.open(os.path.join(self.image_folder, img_id)).convert("RGB")
        references = self.idx2captions[idx]
        # print(f'caption = {caption}')
        # print(f'references = {references}')
        if self.transform is not None:
            img = self.transform(img)

        numericalized_caption = [self.vocab.stoi["<SOS>"]]  # convert word/subword to index
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])

        return (img, torch.tensor(numericalized_caption)), references
