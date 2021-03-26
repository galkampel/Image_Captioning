import argparse
import os
import json
import spacy
import numpy as np
import pandas as pd
from dataset import Vocabulary

# TODO:
#  finish flow (argsparse)
#   i. vocab- save stoi and itos (and load in dataset)
#  ii. CaptionDataset- save transformed df and idx2captions


def get_arguments(arg_list=None):
    parser = argparse.ArgumentParser(description="Model parameters")
    parser.add_argument("--folder_params", type=str, default="input", choices=["input"])
    parser.add_argument("--file_params", type=str, default="train", choices=["preprocess"])
    return parser.parse_args(arg_list)


class Preprocessor:
    def __init__(self, image_folder, caption_file_path, vocab, min_caption_len=3, max_caption_len=35,
                 seed=1):
        self.image_folder = image_folder
        df = self.create_df(caption_file_path, min_caption_len, max_caption_len)
        self.idx2captions = self.create_idx2captions(df['image'].values, df['caption'].values)
        self.df = self.transform_df(df, seed)
        self.imgs = self.df['image']
        self.captions = self.df['caption']
        self.vocab = vocab
        self.vocab.build_vocab(self.captions.tolist())
        self.update_idx2captions()

    @staticmethod
    def create_idx2captions(imgs_name, captions):
        count = 0
        img_prev_name = imgs_name[0]
        idx2captions = {count: [captions[0]]}
        for i, img_name in enumerate(imgs_name[1:], 1):
            if img_name != img_prev_name:
                count += 1
                img_prev_name = img_name
            idx2captions.setdefault(count, []).append(captions[i])
        return idx2captions

    def set_references(self, references):
        tokenized_references = []
        for reference in references:
            tokenized_reference = self.vocab.tokenize(reference)
            tokenized_references.append(["<SOS>"] + tokenized_reference + ["<EOS>"])
        return tokenized_references

    def update_idx2captions(self):
        idx2captions = {}
        for idx, captions in self.idx2aptions.items():
            references = self.set_references(captions)
            idx2captions[idx] = references
        self.idx2captions = idx2captions

    @staticmethod
    def create_df(caption_file, min_caption_len, max_caption_len):
        df = pd.read_csv(caption_file)
        df['caption'] = df['caption'].map(lambda caption: caption[:-2] if caption[-2:] == ' .' else caption)
        df['caption_len'] = df['caption'].map(lambda caption: len(caption.split(' ')))
        df = df.loc[(df['caption_len'] > min_caption_len) & (df['caption_len'] <= max_caption_len), :]
        return df

    # for each image we randomly choose a single caption (from k captions)
    def transform_df(self, df_old, seed):
        np.random.seed(seed)
        imgs_name = df_old['image'].unique()
        captions = []
        idx2captions = self.idx2captions
        for i, img_name in enumerate(imgs_name):
            cand_captions = idx2captions[i]
            idx = np.random.choice(len(cand_captions))
            captions.append(cand_captions[idx])
        df = pd.DataFrame({'image': imgs_name, 'caption': captions})
        return df

    def save_df(self, folder_name='preprocess', filename='df'):
        df_dir = os.path.join(os.getcwd(), folder_name)
        os.makedirs(df_dir, exist_ok=True)
        full_path = os.path.join(df_dir, f'{filename}.csv')
        self.df.to_csv(full_path, index=False)

    def save_dict(self, folder_name='preprocess', filename='vocab'):
        params = None
        dict_dir = os.path.join(os.getcwd(), folder_name)
        os.makedirs(dict_dir, exist_ok=True)
        if filename == 'vocab':
            params = {'stoi': self.vocab.stoi, 'itos': self.vocab.itos}
        elif filename == 'idx2captions':
            params = {filename: self.idx2captions}
        file_path = os.path.join(dict_dir, f'{filename}.json')
        with open(file_path, 'w') as json_file:
            json.dump(params, json_file)


def preprocess(args):
    root_dir = os.getcwd()
    full_path = os.path.join(root_dir, args.folder_params, f'{args.file_params}.json')
    with open(full_path) as json_file:
        preprocess_params = json.load(json_file)
    parser = spacy.load('en_core_web_sm')
    vocab = Vocabulary(parser)
    root_input = os.path.join(root_dir, preprocess_params["input_folder"])  # input_folder = 'flickr8k'
    caption_file = os.path.join(root_input, preprocess_params["caption_file"])  # caption_file='captions.txt'
    img_folder = os.path.join(root_input, preprocess_params["img_folder"])  # img_folder='images'
    preprocessor = Preprocessor(img_folder, caption_file, vocab)
    preprocessor.save_df()
    preprocessor.save_dict(filename='idx2captions')
    preprocessor.save_dict()


if __name__ == '__main__':
    arguments = get_arguments()
    preprocess(arguments)
