import torch
import torch.nn as nn
# import torch.nn.functional as F


class CNNtoRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, **encoder_decoder_params):
        super(CNNtoRNN).__init__()
        encoder_params = encoder_decoder_params["encoder"]
        self.encoder_name = encoder_params.get("model_name", "")
        decoder_params = encoder_decoder_params["decoder"]
        self.decoder_name = decoder_params.get("model_name", "")
        self.encoder = self.set_encoder(embed_size, encoder_params)
        self.decoder = self.set_decoder(vocab_size, embed_size, hidden_size, decoder_params)

    # @staticmethod
    # def set_encoder(embed_size, params):
    #     model_name = params["model_name"]
    #     if model_name == "ViT": # vision transformer
    #         # model = EncoderTransformer()
    #         pass
    #     else: # pretrained CNN model
    #         train_all_model = params.get("train_all_model", False)
    #         return EncoderCNN(embed_size, model_name, train_all_model)
    #
    # @staticmethod
    # def set_decoder(vocab_size, embed_size, hidden_size, params):
    #     model_name = params["model_name"]
    #     if model_name == "transformer":
    #         pass
    #         # return DecoderTransformer()
    #     else:
    #         num_layers = params.get("num_layers", 1)
    #         p = params.get("p", 1.0)
    #         return AttnDecoderRNN(vocab_size, embed_size, hidden_size, num_layers, model_name, p)

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

