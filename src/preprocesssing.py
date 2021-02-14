import os
import pandas as pd
import torch
import spacy
from pandas.tests.resample.conftest import _static_values
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torch.nn.utils.rnn import pad_sequence
from skimage import io
from PIL import Image

# TODO: words and subwords preprocessing
# each image has ~5 corresponding images

class Vocaulary:
    def __init__(self, parser, freq_thres=2):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}  #unk - oov or less than freq_thres
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.parser = parser  # e.g. spacy_eng
        self.freq_thres = freq_thres

    def __len__(self):
        return len(self.itos)

    def tokenizer(self, text):
        return [tok.text.lower for tok in self.parser(text)]

    def build_vocab(self, sentences): #update itos and stoi
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

    def numericalize(self,text):
        tokenized_text = self.parser(text)
        return [self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
                for token in tokenized_text]



class CaptionDataset(Dataset):
    def __init__(self, root, caption_file, vocab, transform=None):  # vocab = Vocabulary(freq_thres)
        self.root = root
        self.df = pd.read_csv(caption_file)
        self.transform = transform
        self.imgs = self.df['image']
        self.captions = self.df['caption']
        self.vocab = vocab
        self.vocab.build_vocab(self.captions.tolist())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        caption = self.captions[idx]
        img_id = self.captions[idx]
        img = Image.open(os.path.join(self.root, img_id)).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        numericalized_caption = [self.vocab.stoi["<SOS>"]]  # convert word/subword to index
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])

        return img, torch.tensor(numericalized_caption)


# padding each batch to be of  the same sequence length
class PadCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs, captions = [] , []
        for img, caption in batch:
            imgs.append(img)
            captions.append(caption)
        imgs = torch.stack(imgs)  # dim=0
        captions = pad_sequence(captions, batch_first=False, padding_value=self.pad_idx)
        return imgs, captions








