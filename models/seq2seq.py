import torch
import torch.nn as nn
import torch.nn.functional as F
from random import random
import numpy as np
# from PIL import Image


class CNNtoRNN(nn.Module):
    def __init__(self, encoder, decoder, device, vocab_size, params):
        super(CNNtoRNN, self).__init__()
        self.encoder = encoder.to(device)
        self.vocab_size = vocab_size
        self.decoder = decoder.to(device)
        # fraction for which we force gold input instead of prediction
        self.teacher_force_ratio = params.get("teacher_force_ratio", 1.0)
        # print(f'teacher force ration = {self.teacher_force_ratio}')
        self.device = device

    def forward(self, imgs, captions):
        features = self.encoder(imgs)  # (B, num_pixels, enc_dim)
        B, max_seq_len = captions.shape  # captions: (B, max_seq_len)
        hiddens = self.decoder.init_hiddens(features)  # (B, (num_layers * num_directions), dec_dim) h_0 (c_0)
        x = captions[:, 0]  # (B) , captions[0] = '<SOS>', captions: (B, max_seq_len)
        # outputs = torch.zeros_like(captions, dtype=torch.int32).to(self.device)  # (B, max_seq_len)
        # outputs = torch.zeros(B, max_seq_len, self.vocab_size).to(self.device)  # (B, max_seq_len, vocab_size)
        outputs = torch.zeros(max_seq_len, B, self.vocab_size).to(self.device)  # (max_seq_len, B, vocab_size)
        # outputs[:, 0] = x
        for i in range(1, max_seq_len):
            logits, hiddens = self.decoder(features, x, hiddens)  # logits: (B, vocab_size)
            # preds = F.log_softmax(logits, dim=1)  # (B,vocab_size)
            # preds = preds.argmax(dim=1)  # (B)
            outputs[i] = logits
            preds = logits.argmax(dim=1)  # (B)
            x = captions[:, i] if random() <= self.teacher_force_ratio else preds  # teacher forcing ratio
            # if i+1 < max_seq_len:
            #     x = captions[i+1] if random() <= self.teacher_force_ratio else preds  # teacher forcing ratio
        return outputs

    def predict(self, img, vocab, **params):  # k- beam size
        # set to eval mode for encoder and decoder
        k = params.get('k', 1)
        max_seq_len = params.get('max_seq_len', 35)
        # self.encoder.eval()
        # self.decoder.attention.eval()
        # self.decoder.eval()
        # with torch.no_grad:
        features = self.encoder(img.unsqueeze(0))  # (1, C, H, W) -> (1, num_pixels, enc_dim)
        x = vocab.stoi["<SOS>"]
        hiddens = self.decoder.init_hiddens(features)
        top_k_tracks = [Tracker([x], hiddens) for _ in range(k)]
        for i in range(1, max_seq_len):
            tmp_tracks = []
            tmp_vals = []
            # for each track from current k best tracks, extend track to k (larger) tracks to get best k tracks
            for j in range(k):
                tracker = top_k_tracks[j]
                if tracker.is_completed():
                    tmp_tracks.append(tracker)
                    continue
                idxes = tracker.get_idxes()
                caption_idx = torch.tensor(idxes[-1]).reshape(1).to(self.device)  # new .to(self.device)
                logits, hiddens, attn_weights = self.decoder(features, caption_idx, tracker.get_prev_hiddens(),
                                                             return_attn_weights=True)
                attn_weights_list = tracker.get_attn_weights_list()
                attn_weights_list.append(attn_weights.view(-1))  # attn_weights: (B=1, num_pixels) -> (num_pixels)
                preds = F.log_softmax(logits, dim=1)  # (B=1,vocab_size)
                top_k_vals, top_k_idxes = torch.topk(preds, k=k, dim=1)  # (B=1, k)
                top_k_vals, top_k_idxes = top_k_vals.tolist()[0], top_k_idxes.tolist()[0]  # k
                # print(f'top_k_idxes:\n{top_k_idxes}')
                # print(f'top_k_vals:\n{top_k_vals}')
                for l in range(k):
                    tmp_val = tracker.calc_new_val(top_k_vals[l])
                    tmp_idx = top_k_idxes[l]
                    # tmp_val = tracker.calc_new_val(top_k_vals[l].item())
                    # tmp_idx = top_k_idxes[l].item()
                    tmp_completed = vocab.itos[tmp_idx] == "<EOS>"
                    tmp_idxes = idxes + [tmp_idx]
                    tmp_tracker = Tracker(tmp_idxes, hiddens, attn_weights_list, tmp_val, tmp_completed)
                    tmp_tracks.append(tmp_tracker)
                    tmp_vals.append(tmp_val)
            cur_k_best_idxes = np.argsort(tmp_vals)[-k:]
            top_k_tracks = [tmp_tracks[idx] for idx in cur_k_best_idxes]
            has_finished = sum([tracker.is_completed() for tracker in top_k_tracks]) == k
            if has_finished:
                break

        best_trackers_value = [tracker.value for tracker in top_k_tracks]
        best_tracker = top_k_tracks[np.argmax(best_trackers_value)]
        best_idxes = best_tracker.get_idxes()
        best_preds = [vocab.itos[idx] for idx in best_idxes]
        best_alphas = best_tracker.get_attn_weights_list()  # T alphas of size (num_pixels)
        best_val = best_tracker.get_value()
        return best_preds, best_alphas, best_val

    def calc_gold_caption_prob(self, img, captions, vocab):  # get caption log probability
        gold_log_prob = 0.0
        # self.encoder.eval()
        # self.decoder.attention.eval()
        # self.decoder.eval()
        T = 0
        features = self.encoder(img.unsqueeze(0))  # (1, C, H, W) -> (1, num_pixels, enc_dim)
        hiddens = self.decoder.init_hiddens(features)
        # x = caption[0]  # (seq_len)  or (1, seq_len)
        # seq_len = len(caption)
        for x in captions:
            logits, hiddens = self.decoder(features, x.reshape(1), hiddens)
            log_probs = F.log_softmax(logits, dim=1)  # (B=1,vocab_size)
            gold_log_prob += log_probs.max().item()
            T += 1
            if vocab.itos[x.item()] == "<EOS>":
                break
        return gold_log_prob / T


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

    # def set_prev_hiddens(self, hiddens):
    #     self.prev_hiddens = hiddens

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
