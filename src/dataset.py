import os
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
from PIL import Image
import numpy as np


def train_test_split(dataset, lengths):
    return random_split(dataset, lengths)


class Vocabulary:
    def __init__(self, parser, freq_thres=2):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}  # unk - oov or less than freq_thres
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.parser = parser  # e.g. spacy_eng
        self.freq_thres = freq_thres

    def __len__(self):
        return len(self.itos)

    def tokenizer(self, text):
        return [tok.text.lower for tok in self.parser(text)]

    def build_vocab(self, sentences):  # update itos and stoi
        freqs = {}
        idx = 4
        for sentence in sentences:
            for word in self.tokenizer(sentence):
                if word not in freqs:
                    freqs[word] = 1
                else:
                    freqs[word] += 1

                if freqs[word] == self.freq_thres:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.parser(text)
        return [self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
                for token in tokenized_text]


# TODO: remember- vocab = Vocabulary(freq_thres)
# each image has ~5 corresponding images
class CaptionDataset(Dataset):
    def __init__(self, root, caption_file, vocab, transform=None, min_caption_len=3, max_caption_len=35, seed=1):
        self.root = root
        df = self.create_df(caption_file, min_caption_len, max_caption_len)
        self.idx2captions = self.create_idx2captions(df['image'].values, df['captions'].values)
        self.df = self.transform_df(df, seed)
        self.transform = transform
        self.imgs = self.df['image']
        self.captions = self.df['caption']
        self.vocab = vocab
        self.vocab.build_vocab(self.captions.tolist())

    @staticmethod
    def create_idx2captions(imgs_name, captions):
        count = 0
        img_prev_name = imgs_name[0]
        idx2captions = {count: [captions[0].split()]}
        for i, img_name in enumerate(imgs_name[1:], 1):
            if img_name != img_prev_name:
                count += 1
            idx2captions.get(count, []).append(captions[i].split())
        return idx2captions

    def create_df(self, caption_file, min_caption_len, max_caption_len):
        # remove sentences that end with ' .' and short sentences (less than min_caption_len)
        df = pd.read_csv(caption_file)
        df['caption'] = df['caption'].map(lambda caption: caption[:-2] if caption[-2:] == ' .' else caption)
        df['caption_len'] = df['caption'].map(lambda caption: len(caption.split(' ')))
        df = df.loc[(df['caption_len'] > min_caption_len) & (df['caption_len'] > max_caption_len), :]

        return df

    # for each image we randomly choose a single caption (from k captions)
    def transform_df(self, df_old, seed):
        np.random.seed(seed)
        imgs_name = df_old['image'].unique()
        captions = []
        idx2captions = self.idx2captions
        for i, img_name in enumerate(imgs_name):
            cand_captions = idx2captions[i]
            caption = np.random.choice(cand_captions)
            captions.append(caption)
        df = pd.DataFrame({'image': imgs_name, 'caption': captions})
        return df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        caption = self.captions[idx]
        img_id = self.imgs[idx]
        img = Image.open(os.path.join(self.root, img_id)).convert("RGB")
        references = self.idx2captions[idx]
        if self.transform is not None:
            img = self.transform(img)

        numericalized_caption = [self.vocab.stoi["<SOS>"]]  # convert word/subword to index
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])

        return (img, torch.tensor(numericalized_caption)), references
