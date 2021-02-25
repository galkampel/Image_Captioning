import torch
import torch.nn as nn
import torch.nn.functional as F
from random import random
import numpy as np
# from PIL import Image
from skimage import transform as s_transform
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os


class CNNtoRNN(nn.Module):
    def __init__(self, encoder, decoder, device, teacher_force_ratio=1.0):
        super(CNNtoRNN).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.teacher_force_ratio = teacher_force_ratio
        self.device = device

    def forward(self, imgs, captions):
        features = self.encdoer(imgs)
        max_seq_len = captions.shape[1]  # (B, max_seq_len)
        hiddens = self.decoder.init_hiddens(features)  # (B, (num_layers * num_directions * decoder_features)) h_0 (c_0)
        x = captions[:, 0]  # (B) , captions[0] = '<SOS>'
        outputs = torch.zeros_like(captions, dtype=torch.int32).to(self.device)
        for i in range(max_seq_len):
            logits, hiddens = self.decoder(features, x, hiddens)
            preds = F.log_softmax(logits, dim=1)  # (B,vocab_size)
            preds = preds.argmax(dim=1)
            outputs[:, i] = preds
            if i+1 < max_seq_len:
                x = captions[i+1] if random() <= self.teacher_force_ratio else preds  # teacher forcing ratio
        return outputs

    def evaluate(self, img, vocab, k=1, max_seq_len=35):  # k- beam size
        with torch.no_grad:
            features = self.encdoer(img.unsqueeze(0))  # (1, C, H, W) -> (1, num_pixels, enc_dim)
            x = vocab.stoi["<SOS>"]
            hiddens = self.decoder.init_hiddens(features)
            top_k_tracks = [Tracker([x], hiddens) for _ in range(k)]
            for i in range(1, max_seq_len):
                tmp_tracks = []
                tmp_vals = []
                for j in range(k):
                    tracker = top_k_tracks[j]
                    if tracker.is_completed():
                        tmp_tracks.append(tracker)
                        continue
                    idxes = tracker.get_idxes()
                    logits, hiddens, attn_weights = self.decoder(features, idxes[-1], tracker.get_prev_hiddens(),
                                                                 return_attn_weights=True)
                    attn_weights_list = tracker.get_attn_weights_list()
                    attn_weights_list.append(attn_weights.view(-1))  # attn_weights: (B=1, num_pixels) -> (num_pixels)
                    preds = F.log_softmax(logits, dim=1)  # (B=1,vocab_size)
                    top_k_vals, top_k_idxes = torch.topk(preds, k=k, dim=1)  # (B=1, k)
                    top_k_vals, top_k_idxes = top_k_vals.tolist()[0], top_k_idxes.tolist()[0]  # k
                    for l in range(k):
                        tmp_val = tracker.calc_new_val(top_k_vals[l].item())
                        tmp_idx = top_k_idxes[l].item()
                        tmp_completed = vocab.itos(tmp_idx) == "<EOS>"
                        tmp_idxes = idxes + [tmp_idx]
                        tmp_tracker = Tracker(tmp_idxes, hiddens, attn_weights_list, tmp_val, tmp_completed)
                        tmp_tracks.append(tmp_tracker)
                        tmp_vals.append(tmp_val)
                cur_best_idx = np.argsort(tmp_val)[-k:]
                top_k_tracks = [tmp_tracks[idx] for idx in cur_best_idx]
                has_finished = sum([tracker.is_completed() for tracker in top_k_tracks]) == k
                if has_finished:
                    break

            best_trackers_value = [tracker.value for tracker in top_k_tracks]
            best_tracker = top_k_tracks[np.argmax(best_trackers_value)]
            best_idxes, best_val = best_tracker.get_idxes(), best_tracker.get_value()
            best_preds = [vocab.itos[idx] for idx in best_idxes]
            best_alphas = best_tracker.get_attn_weights_list()  # T alphas of size (num_pixels)
        return best_preds, best_val, best_alphas

    def calc_gold_caption_prob(self, img, caption, vocab):  # get caption log probability
        gold_log_prob = 0.0
        with torch.no_grad:
            features = self.encdoer(img.unsqueeze(0))  # (1, C, H, W) -> (1, num_pixels, enc_dim)
            hiddens = self.decoder.init_hiddens(features)
            x = caption[0]  # (seq_len)  or (1, seq_len)
            seq_len = len(caption)
            for i in range(1, seq_len):
                logits, hiddens = self.decoder(features, x, hiddens)
                log_probs = F.log_softmax(logits, dim=1)  # (B=1,vocab_size)
                gold_log_prob += log_probs.max().item()
                if vocab.itos[x] == "<EOS>":
                    break
                x = caption[i]

        return gold_log_prob

    def visualize_caption(self, img, img_name, vocab, enc_dim=16, img_dim=224, means=(0.485, 0.456, 0.406),
                          sds=(0.229, 0.224, 0.225), k=1, max_seq_len=35, smooth=True, img_dir='img'):
        caption, _, attn_weights_lst = self.evaluate(img, vocab, k, max_seq_len)
        # unnormalize to in range 0-255: z = (x- mu) / sigma -> x = z - (-mu/sigma) / 1/sigma
        reveresed_means, reversed_sds = zip(*[(-means[i] / sd, 1 / sd) for i, sd in enumerate(sds)])
        inv_transform = transforms.Compose([
            transforms.Normalize(reveresed_means, reversed_sds),
            transforms.ToPILImage()
        ])
        pil_img = inv_transform(img)
        # image = pil_img.resize([14 * 24, 14 * 24], Image.LANCZOS)
        N = len(caption)
        upscale = img_dim // enc_dim
        for t, word in enumerate(caption):
            plt.subplot(np.ceil(N / 5.), 5, t + 1)  # (N // 5)X5 subplots.
            plt.text(0, 1, word, color='black', backgroundcolor='white', fontsize=10)
            plt.imshow(pil_img)
            attn_weights = attn_weights_lst[t].view(enc_dim, enc_dim)  # (num_pixels) -> (enc_dim, enco_dim)
            if smooth:
                attn_weights = s_transform.pyramid_expand(attn_weights.numpy(), upscale=upscale, sigma=8)
            else:
                attn_weights = s_transform.resize(attn_weights.numpy(), [enc_dim * upscale, enc_dim * upscale])

            if t == 0:
                plt.imshow(attn_weights, alpha=0)
            else:
                plt.imshow(attn_weights, alpha=0.8)
            plt.set_cmap(cm.Greys_r)
            plt.axis('off')
        par_dir = os.path.abspath(os.path.join(os.getcwd(), os.path.pardir))
        img_folder = os.path.join(par_dir, img_dir)
        os.makedirs(img_folder, exist_ok=True)
        img_path = os.path.join(img_folder, f'{img_name}.png')
        plt.savefig(img_path)
        plt.close()


class Tracker:
    def __init__(self, idxes=None, prev_hiddens=None, attn_weights_lst=None,  value=-np.inf, completed=False):
        self.idxes = [idx for idx in idxes] if idxes is not None else []
        self.value = value
        self.completed = completed
        self.prev_hiddens = prev_hiddens
        self.attn_weights_lst = attn_weights_lst if attn_weights_lst is not None else []

    def get_idxes(self):
        return [idx for idx in self.idxes]

    def get_value(self):
        return self.value

    def get_prev_hiddens(self):
        return self.prev_hiddens

    def get_attn_weights_list(self):
        return [attn_weight for attn_weight in self.attn_weights_lst]

    def set_prev_hiddens(self, hiddens):
        self.prev_hiddens = hiddens

    def is_completed(self):
        return self.completed

    def set_status(self, completed):
        self.completed = completed

    def calc_new_val(self, log_prob):
        T = len(self.idxes) - 1  # ignore <SOS>
        if T == 0:
            return log_prob
        else:
            return (T / (T+1)) * (self.value + log_prob)
