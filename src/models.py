import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class EncoderCNN(nn.Module):
    def __init__(self, embed_size, model_name='resnet50', train_all_model=False):
        super(EncoderCNN).__init__()
        if model_name == 'resnet50':
            self.pretrained_model = models.resnet50(pretrained=True)
        self.pretrained_model.fc = nn.Linear(self.model.fc.in_features, embed_size)
        if not train_all_model:
            self.freeze_model_weights()

    def freeze_model_weights(self):
        for name, params in self.pretrained_model.named_parameters():
            if "fc.weight" not in name and "fc.bias" not in name:
                params.requires_grad = False

    def forward(self, x):
        x = self.pretrained_model(x)
        x = F.relu(x)
        return x


class EncoderTransformer(nn.Module): # ViT
    def __init__(self, model_name):
        super(EncoderTransformer).__init__()

    def forward(self, x):
        pass


class DecoderRNN(nn.Module):  # lstm/gru decoder
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, model_name, p=0.0):
        super(DecoderRNN).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.model_name = model_name
        if self.model_name == 'lstm':
            self.rnn = nn.LSTM(embed_size, hidden_size, num_layers)
        elif self.model_name == 'gru':
            self.rnn = nn.GRU(embed_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(p)

    def forward(self, features, captions):
        embeddings = self.dropout(self.embedding(features))  # x- a concat of CNN feature maps and the caption sequence
        rnn_input = torch.cat((embeddings.unsqueeze(0), captions), dim=0)
        rnn_output, _ = self.rnn(rnn_input) # if h0 ((h0,c0)) are not given = 0 set to 0, output ys and h_n ((h_n,c_n))
        out = self.fc(rnn_output)
        return out


class DecoderTransformer(nn.Module):  # transformer as a decoder
    def __init__(self):
        super(DecoderTransformer).__init__()
        pass

    def forward(self, x):
        pass


class CNNtoRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, **encoder_decoder_params):
        super(CNNtoRNN).__init__()
        encoder_params = encoder_decoder_params["encoder"]
        self.encoder_name = encoder_params.get("model_name", "")
        decoder_params = encoder_decoder_params["decoder"]
        self.decoder_name = decoder_params.get("model_name", "")
        self.encoder = self.set_encoder(embed_size, encoder_params)
        self.decoder = self.set_decoder(vocab_size, embed_size, hidden_size, decoder_params)

    @staticmethod
    def set_encoder(embed_size, params):
        model_name = params["model_name"]
        if model_name == "ViT": # vision transformer
            # model = EncoderTransformer()
            pass
        else: # pretrained CNN model
            train_all_model = params.get("train_all_model", False)
            return EncoderCNN(embed_size, model_name, train_all_model)

    @staticmethod
    def set_decoder(vocab_size, embed_size, hidden_size, params):
        model_name = params["model_name"]
        if model_name == "transformer":
            pass
            # return DecoderTransformer()
        else:
            num_layers = params.get("num_layers", 1)
            p = params.get("p", 1.0)
            return DecoderRNN(vocab_size, embed_size, hidden_size, num_layers, model_name, p)

    def forward(self, imgs, captions):
        features = self.encdoer(imgs)
        outputs = self.decoder(features, captions)
        return outputs

    def get_caption(self, img, vocab, max_length=50):  # get predicted caption
        caption_idxes = []
        with torch.no_grad:
            x = self.encoder(img.unsqueeze(0)) # or self.encoder(img).unsqueeze(0)
            states = None
            for _ in range(max_length):

                if self.encoder_name == "tansformer":
                    pass
                else:
                    hiddens, states = self.decoder.rnn(x, states)
                    logits = self.decoder.fc(hiddens.squeeze(0))
                    pred = logits.argmax(1)
                    caption_idxes.append(pred.item())
                    x = self.decoder.embedding(pred).unsqueeze(0)

                if vocab.itos[pred.item()] == "<EOS>":
                    break
        return [vocab.itos[idx] for idx in caption_idxes]

