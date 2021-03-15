import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, enc_dim, dec_dim, params):
        super(Attention).__init__()
        attn_type = params.get('type', 'additive')
        self.attn_type = attn_type
        if self.attn_type == 'multiplicative':
            self.W = nn.Linear(dec_dim, enc_dim)
        elif self.attn_type == 'additive':
            attn_dim = params.get('attention_dim', 256)
            self.W1 = nn.Linear(enc_dim, attn_dim)
            self.W2 = nn.Linear(dec_dim, attn_dim)
            self.V = nn.Linear(attn_dim, 1)
            # self.V = nn.Parameter(nn.init.normal_(self.V))  # for learned vector (requires_grad=True as default)

    def set_attn_score(self, enc_hiddens, dec_prev_hidden):
        """
        Args:
            enc_hiddens: encoder's features (B, num_pixels, enc_hidden_size)
            dec_prev_hidden: decoder's previous (first) hidden layer (B, dec_hidden_size)

        Returns:
            score: attention score  # (B, num_pixels)
        """
        scores = None
        if self.attn_type == 'multiplicative':  # s^T* W * h_i , s- decoder, h- encoder
            h_prev_projected = self.W(dec_prev_hidden).unsqueeze(-1)  # (B, enc_dim, 1)
            scores = torch.matmul(enc_hiddens, h_prev_projected).squeeze(-1)  # (B, num_pixels, 1) -> (B, num_pixels)
        elif self.attn_type == 'additive':  # tanh(W_1h_i + W_2s ) V^T
            h_prev_projected = self.W1(dec_prev_hidden)  # (B, attn_dim)
            enc_hiddens_projected = self.W2(enc_hiddens)  # (B, num_pixels, attn_dim)
            scores = torch.tanh(h_prev_projected.unsqueeze(1) + enc_hiddens_projected)  # (B, num_pixels, attn_dim)
            scores = self.V(scores).squeeze(-1)  # (B, num_pixels)
        elif self.attn_type == 'basic':  # apply basic only if enc_dim=dec_dim
            scores = torch.matmul(enc_hiddens, dec_prev_hidden.unsqueeze(1)).squeeze(-1)  # (B, num_pixels)
        return scores

    def forward(self, enc_hiddens, dec_prev_hidden):
        """
        Args:
            enc_hiddens: encoder's hidden states (features)  (B, num_pixels, enc_dim)
            dec_prev_hidden: decoder previous hidden state (B, hidden_size)/(B,(num_layers*num_directions), hidden_size)
            h_n.view(num_layers, num_directions, batch, hidden_size)
        Returns:
            context vector (B, enc_dim) and attention weights (B, num_pixels)
        """
        attn_score = self.set_attn_score(enc_hiddens, dec_prev_hidden)  # (B, num_pixels)
        attn_weights = F.softmax(attn_score, dim=1)  # (B, num_pixels)
        context_vectors = (attn_weights.unsqueeze(-1) * enc_hiddens).sum(axis=1).unsqueez(1)  # (B, seq_len=1, enc_dim)
        return context_vectors, attn_weights


class AttnDecoderRNN(nn.Module):  # lstm/gru decoder
    def __init__(self, attention, vocab_size, enc_hidden_size, dec_hidden_size, params):
        super(AttnDecoderRNN).__init__()
        self.model_name = params.get('model_name', 'lstm')
        self.num_layers = params.get('num_layers', 2)
        self.num_directions = params.get('num_directions', 1)
        self.embed_size = params.get('embed_size', 256)
        self.embedding = nn.Embedding(vocab_size, self.embed_size)
        p = params.get('dropout', 0.0)  # no dropout
        self.W_h = nn.Linear(enc_hidden_size, self.num_layers * self.num_directions * dec_hidden_size)
        if self.model_name == 'lstm':
            self.W_c = nn.Linear(enc_hidden_size, self.num_layers * self.num_directions * dec_hidden_size)
            self.rnn = nn.LSTM(self.embed_size + enc_hidden_size, dec_hidden_size, self.num_layers, batch_first=True,
                               dropout=p,
                               bidirectional=bool(self.num_directions - 1))  # (B, seq_len, features)
        elif self.model_name == 'gru':
            self.rnn = nn.GRU(self.embed_size + enc_hidden_size, dec_hidden_size, self.num_layers, batch_first=True,
                              dropout=p, bidirectional=bool(self.num_directions - 1))  # (B, seq_len, features)
        self.attention = attention
        self.fc = nn.Linear(dec_hidden_size * self.num_directions, vocab_size)
        self.dropout = nn.Dropout(p)

    def init_hiddens(self, enc_features):
        B = enc_features.shape[0]
        avg_features = enc_features.mean(dim=1)  # (B, num_pixels, enc_dim) -> (B, enc_dim)
        h = self.W_h(avg_features)  # (B, encoder_features) -> (B, (num_layers * num_directions * dec_dim))
        h = h.view(B, self.num_layers * self.num_directions, -1)  # (B, (num_layers * num_directions), dec_dim)
        h = F.relu(h)  # maybe tanh instead (torch.tanh(h))
        if self.model_name == 'lstm':
            c = self.W_c(avg_features)  # (B, (num_layers * num_directions * decoder_features))
            c = c.view(B, self.num_layers * self.num_directions, -1)  # (B, (num_layers * num_directions), dec_dim)
            c = F.relu(c)  # maybe tanh instead (torch.tanh(c))
            return h, c
        return h

    def forward(self, features, captions, hiddens, return_attn_weights=False):
        """
        Args:
            features: encoder's features (B, num_pixels, enc_dim)
            captions: a single word in a (B)
            hiddens: previous hidden states (if lstm h_{t-1} and c_{t-1}) (B, (1*num_layers*num_directions), dec_dim)
            return_attn_weights: if true returns the attention weights as well (for visualization)

        Returns:
            out: rnn's output, i.e.the logits  (B, vocab_size)
            hiddens: current hidden states (if lstm h_{t-1} and c_{t-1}) (B, (1*num_layers*num_directions), dec_dim)
        """
        embeddings = self.dropout(self.embedding(captions))  # (B, embed_size)
        h_prev = hiddens if self.model_name == 'gru' else hiddens[0]  # (B, num_layers * num_directions, dec_dim)
        if self.num_layers > 1:  # assumes num_directions = 1
            h_prev = h_prev.view(-1, self.num_layers, self.embed_size)[:, 0, :]  # (B, dec_dim)
        context_vecs, attn_weights = self.attention(features, h_prev)
        rnn_input = torch.cat((embeddings.unsqueeze(1), context_vecs), dim=-1)  # (B, seq_len=1, (embed_dim + enc_dim))
        rnn_output, hiddens = self.rnn(rnn_input, hiddens)  # h_i or (h_i,c_i) if lstm
        out = self.fc(rnn_output.squeeze(1))  # (B, seq_len=1, num_directions=1 * hidden_size) -> out: (B, vocab_size)
        if return_attn_weights:
            return out, hiddens, attn_weights
        return out, hiddens
