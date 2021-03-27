import os
import numpy as np
import torch
import torch.nn as nn
from models.encoder import EncoderCNN
from models.decoder import AttnDecoderRNN, Attention
from models.seq2seq import CNNtoRNN
from torch.optim import Adam
from torchtext.data import metrics
from skimage import transform as s_transform
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# TODO:
#  save and load model
#  visualization
#  get caption log probability (convert with vocab outside the words/subwords) (done?)
#  make sure that all models are .to(device) (done?)
#  create a different optimizer for encoder (if needed) and decoder (done)
#  beam search analysis true caption vs. predicted caption (done)
#  evaluate bleu (given preds, and references, max_len=4) (done)


class Trainer:
    def __init__(self, criterion, device, vocab_size, enc_dim=2048, dec_dim=512, ** models_params):
        super().__init__()
        self.device = device
        self.criterion = criterion
        self.encoder = EncoderCNN(models_params['encoder'])
        self.vocab_size = vocab_size
        # encoder params: model_name=resnet50, feature_size=16, train_all_model=False,
        # unfreeze_params: {'tune': True, 'epoch': 13, 'layers_name': ['layer_4'?]}
        self.attention = Attention(enc_dim, dec_dim, models_params['attention']).to(device)
        # attention_params: 'type'='additive', 'attention_dim'=256
        self.decoder = AttnDecoderRNN(self.attention, vocab_size, enc_dim, dec_dim, models_params['attention'])
        # decoder params: 'embed_size'=512, 'model_name'='lstm', 'num_layers'=2, ''num_directions'=1, 'p'=0.0
        self.model = CNNtoRNN(self.encoder, self.decoder, device, vocab_size, models_params['seq2seq'])
        self.model = self.model.to(device)
        self.has_unfroze_encoder = False
        # seq2seq params: 'teacher_force_ratio'= 1.0 or 0.5

    def unfreeze_encoder_weights(self, epoch):
        unfreeze_flag = self.model.encoder.unfreeze_model_weights(epoch)
        if unfreeze_flag:
            self.has_unfroze_encoder = unfreeze_flag

    def set_optimizer(self, component_name, optimizer_params):
        lr = optimizer_params.get('lr', 1e-3)
        wd = optimizer_params.get('wd', 0.0)
        optimizer_func = optimizer_params.get('optimizer_func', Adam)
        optimizer = None
        if component_name == 'decoder':
            optimizer = optimizer_func(self.decoder.parameters(), lr=lr, weight_decay=wd)
        elif component_name == 'encoder':
            optimizer = optimizer_func(
                params=filter(lambda p: p.requires_grad, self.encoder.parameters()), lr=lr, weight_decay=wd)
        return optimizer

    def set_model_mode(self, train_mode=False):
        if train_mode:
            self.model.train()
            self.model.encoder.train()
            self.model.decoder.train()
            self.model.decoder.attention.train()
        else:
            self.model.eval()
            self.model.encoder.eval()
            self.model.decoder.eval()
            self.model.decoder.attention.eval()

    def train(self, train_loader, optimizer_dec, optimizer_enc=None, clip_dec=5.0, clip_enc=5.0):
        # return the average training loss of an epoch
        self.set_model_mode(train_mode=True)
        total_loss = 0.0
        num_examples = 0
        for (imgs, captions), _ in train_loader:
            imgs, captions = imgs.to(self.device), captions.to(self.device)  # captions: (B, seq_len) -> (B*seq_len)
            optimizer_dec.zero_grad()
            if optimizer_enc is not None:
                optimizer_enc.zero_grad()
            preds = self.model(imgs, captions)  # (max_seq_len, B, vocab_size)
            # print(f"preds' shape before:\n{preds.shape}")
            # print(f'captions dtype:\n{captions.dtype}')
            # print(f'preds dtype:\n{preds.dtype}')
            # print(f'preds shape before view:\n{preds.shape}')
            preds = preds[1:].permute(1, 0, 2).contiguous().view(-1, self.vocab_size)
            # print(f'preds shape after view:\n{preds.shape}')
            loss = self.criterion(preds, captions[:, 1:].contiguous().view(-1))
            B = imgs.size(0)
            total_loss += (loss.item() * B)  # sum loss over all examples
            num_examples += B
            loss.backward()
            # gradient clipping
            nn.utils.clip_grad_value_(self.decoder.parameters(), clip_value=clip_dec)
            optimizer_dec.step()
            if optimizer_enc is not None:
                # gradient clipping
                nn.utils.clip_grad_value_(filter(lambda p: p.requires_grad, self.encoder.parameters()),
                                          clip_value=clip_enc)
                optimizer_enc.step()
        total_loss /= num_examples
        return total_loss

    def evaluate_example(self, img, references, vocab, beam_search_params):  # evaluate one example (bleu score)
        # beam search params: k (1, 3), max_seq_len = 35
        caption_pred, alphas, pred_val = self.model.predict(img, vocab, **beam_search_params)
        # print(f"captions' predictions:\n{caption_pred}")
        # print(f"references:\n{references}")

        return caption_pred, pred_val

    # def beam_search_analysis(self, img, caption, vocab, pred_val):  # check beam search on validation set
    #     gold_val = self.model.calc_gold_caption_prob(img, caption, vocab)
        # print(f'gold caption value = {gold_val}\npredicted caption value = {pred_val}')
        # if gold_val < pred_val:
        #     print('model may not be rich enough')
        # else:
        #     print('k may be too small')
        # return golde_val

    def evaluate(self, data_loader, vocab, beam_search_params, apply_beam_analysis=False):
        self.set_model_mode(train_mode=False)
        # bleu_scores = 0.0
        n_examples = 0
        pred_caption_lst = []
        gold_caption_lst = []
        gold_val_lst = []
        pred_val_lst = []
        with torch.no_grad():
            for (imgs, captions), references in data_loader:
                imgs, captions = imgs.to(self.device), captions.to(self.device)
                B = imgs.size(0)
                n_examples += B
                for i in range(B):
                    pred_caption, pred_val = self.evaluate_example(imgs[i], references[i], vocab, beam_search_params)
                    pred_caption_lst.append(pred_caption)
                    gold_caption_lst.append(references[i])
                    # bleu_scores += pred_bleu
                    if apply_beam_analysis:
                        # self.beam_search_analysis(imgs[i],  captions[i], vocab, pred_val)
                        gold_val = self.model.calc_gold_caption_prob(imgs[i], captions[i], vocab)
                        pred_val_lst.append(pred_val)
                        gold_val_lst.append(gold_val)
        if apply_beam_analysis:
            mean_gold_val = np.mean(gold_val_lst)
            mean_pred_val = np.mean(pred_val_lst)
            print(f"mean gold captions' value = {mean_gold_val}\n mean predicted captions' value = {mean_pred_val}")
            if mean_gold_val < mean_pred_val:
                print('model may not be rich enough')
            else:
                print('k may be too small')
        # pred_bleu = bleu_score(caption_pred, references)  # max n_gram=4
        # bleu_scores /= n_examples
        bleu_score = metrics.bleu_score(pred_caption_lst, gold_caption_lst)
        return bleu_score

    def visualize_caption(self, img, img_name, vocab, enc_dim=16, img_dim=224, means=(0.485, 0.456, 0.406),
                          sds=(0.229, 0.224, 0.225), k=1, max_seq_len=35, smooth=True, img_dir='images'):
        caption, attn_weights_lst, _ = self.model.predict(img, vocab, k=k, max_seq_len=max_seq_len)
        # unnormalize to in range 0-255: z = (x- mu) / sigma -> x = z - (-mu/sigma) / 1/sigma
        reveresed_means, reversed_sds = zip(*[(-means[i] / sd, 1 / sd) for i, sd in enumerate(sds)])
        inv_transform = transforms.Compose([
            transforms.Normalize(reveresed_means, reversed_sds),
            transforms.ToPILImage()
        ])
        # convert tensor image to pil image
        pil_img = inv_transform(img)
        # image = pil_img.resize([14 * 24, 14 * 24], Image.LANCZOS)
        N = len(caption)
        upscale = img_dim // enc_dim  # enc_dim = 16/14
        for t, word in enumerate(caption):
            plt.subplot(np.ceil(N / 5.), 5, t + 1)  # (N // 5)X5 subplots.
            plt.text(0, 1, word, color='black', backgroundcolor='white', fontsize=10)
            plt.imshow(pil_img)
            if t == 0:
                continue
            attn_weights = attn_weights_lst[t].view(enc_dim, enc_dim)  # (num_pixels) -> (enc_dim, enc_dim)
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

    def load_model(self, optimizer_dec, optimizer_enc=None, **params):  # model_name, folder_checkpoint
        checkpoint_folder = params.get('checkpoint_folder', 'checkpoint')
        model_name = params.get('model_name', 'resnet_lstm')
        full_path = os.path.join(checkpoint_folder, f'{model_name}.pt')
        checkpoint = torch.load(full_path)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.attention.load_state_dict(checkpoint['attention_state_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        self.decoder.attention = self.attention
        optimizer_dec.load_state_dict(checkpoint['decoder_optimizer_dict'])
        epoch = checkpoint['epoch']
        if optimizer_enc is not None:
            optimizer_enc.load_state_dict(checkpoint['encoder_optimizer_dict'])
        bleu_score_dict = checkpoint["bleu_score_dict"]
        return optimizer_dec, optimizer_enc, bleu_score_dict, epoch

    # save encoder, decoder, attention and optimizers
    def save_model(self, epoch, optimizer_dec, optimizer_enc, bleu_score_dict, **params):
        checkpoint_folder = os.path.join(os.getcwd(), params.get('checkpoint_folder', 'checkpoint'))
        os.makedirs(checkpoint_folder, exist_ok=True)
        model_name = params.get('model_name', 'resnet_lstm')
        # folder_checkpoint, model_name
        model_saved_name = model_name + f'_epoch={epoch}'
        # print(f'model filename = {model_saved_name}')
        full_path = os.path.join(checkpoint_folder, f'{model_saved_name}.pt')
        params = {'encoder_state_dict': self.model.encoder.state_dict(),
                  'decoder_state_dict': self.model.decoder.state_dict(),
                  'attention_state_dict': self.model.decoder.attention.state_dict(),
                  'decoder_optimizer_dict': optimizer_dec.state_dict(),
                  'bleu_score_dict': bleu_score_dict,
                  'epoch': epoch
                  }
        if optimizer_enc is not None:
            params['encoder_optimizer_dict'] = optimizer_enc.state_dict()
        torch.save(params, full_path)
