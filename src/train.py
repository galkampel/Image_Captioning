
from models.encoder import *
from models.decoder import *
from torchtext.data.metrics import bleu_score

# TODO: beam search analysis true caption vs. predicted caption
# TODO: evaluate bleu (given preds, and references, max_len=4)
# TODO: get caption log probability (convert with vocab outside the words/subwords)


class Trainer:
    def __init__(self, optimizer, models_name, device):
        self.optimizer = optimizer
        self.device = device
        self.seq2seq = self.init_models()

    def init_models(self, models_name):
        encoder = EncoderCNN(model_name, encodrer_dim, train_all_model)

    def train(self, train_loader):
        for imgs, captions in train_loader:
            imgs, captions = imgs.to(self.device), captions.to(self.device)
            self.optimizer.zero_grad()

            loss.backward()
            self.optimizer.step()

    def evaluate(self, loader):
        pass

    def visualize(self):
        pass

    def load_model(self):
        pass

    def save_model(self):
        pass




