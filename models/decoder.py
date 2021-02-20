import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attn_type='basic', attn_dim=256):  # can apply basic if enc_dim=dec_dim
        self.attn_type = attn_type
        if self.attn_type == 'multiplicative':
            self.W = nn.Linear(decoder_dim, encoder_dim)
        elif self.attn_type == 'additive':
            self.W1 = nn.Linear(attn_dim, encoder_dim)
            self.W2 = nn.Linear(attn_dim, decoder_dim)
            self.V = torch.empty((1, attn_dim))
            self.V = nn.Parameter(nn.init.normal_(self.V))  # requires_grad=True


    def set_attn_score(self, enc_hiddens, dec_prev_hidden):
        if self.attn_type == 'multiplicative':
            pass  # s^T* W * h_i , s- decoder, h- encoder
        elif self.attn_type == 'additive':
            pass  # v^T tanh(W_1h_i + W_2s )

        return 0

    def forward(self, enc_hiddens, dec_prev_hidden):  #enc_hiddens = (BXnum_pixelsXenc_dim) dec_hidden = (BX1*num_layers*num_directionsX
        attn_score = self.set_attn_score(enc_hiddens, dec_prev_hidden)
        attn_weights = F.softmax(attn_score, dim=1)  #(BXnum_pixels)
        context_vectors = ...
        return context_vectors, attn_weights


class AttnDecoderRNN(nn.Module):  # lstm/gru decoder
    def __init__(self, model_name, vocab_size, embed_size, enc_hidden_size, dec_hidden_size, num_layers,
                 num_directions=1, p=0.0):  # embed_size = 512, embed_size = 512
        super(AttnDecoderRNN).__init__()
        self.model_name = model_name
        self.num_layers = num_layers
        self.num_directions = num_directions
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.init_h = nn.Linear(enc_hidden_size, num_layers * num_directions * dec_hidden_size)
        if self.model_name == 'lstm':
            self.init_c = nn.Linear(enc_hidden_size, num_layers * num_directions * dec_hidden_size)
            self.rnn = nn.LSTM(embed_size + enc_hidden_size, dec_hidden_size, num_layers, batch_first=True, dropout=p,
                               bidirectional=bool(num_directions - 1)) # BXseq_lenXfeatures
        elif self.model_name == 'gru':
            self.rnn = nn.GRU(embed_size + enc_hidden_size, dec_hidden_size, num_layers, batch_first=True, dropout=p,
                              bidirectional=bool(num_directions - 1)) # BXseq_lenXfeatures
        self.fc = nn.Linear(dec_hidden_size, vocab_size)
        self.dropout = nn.Dropout(p)

    def init_hiddens(self, enc_features):
        avg_features = enc_features.mean(dim=1)  # (BXnum_pixelsXencoder_features) -> (BXencoder_features)
        h = self.init_h(avg_features) # (BXencoder_features) -> (BXnum_layers * num_directions*decoder_features)
        B = enc_features.shape[0]
        h = h.view(B, self.num_layers * self.num_directions, -1)  # (BX(num_layers * num_directions*decoder_features)) -> (BX(num_layers * num_directions)Xdecoder_features)
        h = F.relu(h)  # maybe tanh instead (torch.tanh(h))
        if self.model_name == 'lstm':
            c = self.init_c(avg_features)
            c = c.view(B, self.num_layers * self.num_directions, -1)
            c = F.relu(h)  # maybe tanh instead (torch.tanh(c))
            return h, c
        return h



    def forward(self, features, captions):
        embeddings = self.dropout(self.embedding(captions))  # (BXmax_seq_lenXembed_size)
        B, seq_len, _ = embeddings.shape
        hiddens = self.init_hiddens(features)
        context_vec = ...
        # take the first-layer decoder (previous) hidden layer
        rnn_input = torch.cat((embeddings[0].unsqueeze(0), context_vec), dim=2)

        for i in range(1, seq_len):
            rnn_output, hiddens = self.rnn(rnn_input, hiddens)  # h_i or (h_i,c_i) if lstm
        out = self.fc(rnn_output)
        return out


class DecoderTransformer(nn.Module):  # transformer as a decoder
    def __init__(self):
        super(DecoderTransformer).__init__()
        pass

    def forward(self, x):
        pass

