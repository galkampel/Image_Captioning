# import torch
import torch.nn as nn
import torchvision.models as models
# import torch.nn.functional as F
from collections import OrderedDict


class EncoderCNN(nn.Module):
    def __init__(self, model_name='resnet50', encoder_dim=16, train_all_model=False, unfreeze_params=None):
        super(EncoderCNN).__init__()
        if model_name == 'resnet50':
            resnet50 = models.resnet50(pretrained=True)  # need to remove adaptive avg. pooling + fc layer
            self.pretrained_model = nn.Sequential(OrderedDict([
                (name, params) for name, params in list(resnet50.named_children())[:-2]]))

        self.adaptavgpool2d = nn.AdaptiveAvgPool2d(encoder_dim)
        if not train_all_model:
            self.freeze_model_weights()

        self.unfreeze_params = unfreeze_params if unfreeze_params is not None else {}

    def freeze_model_weights(self):
        for name, params in self.pretrained_model.named_parameters():
            if "fc.weight" not in name and "fc.bias" not in name:
                params.requires_grad = False

    def unfreeze_weights(self, layers_name):
        i = 0
        for layer_name, child_params in self.pretrained_model.named_children():
            if layer_name == layers_name[i]:
                for params in child_params.parameters():
                    params.requires_grad = True
                i += 1
                if i >= len(layers_name):
                    break

    def unfreeze_model_weights(self, epoch):  # unfreeze parts of pretrained model from a specific epoch
        if self.unfreeze_params.get('tune', False):
            if self.unfreeze_params.get('epoch', 1) <= epoch:
                layers_name = self.unfreeze_params.get('layer', [])
                if len(layers_name) > 0:
                    self.unfreeze_weights(layers_name)
                self.unfreeze_params['tune'] = False

    def forward(self, x):
        x = self.pretrained_model(x)
        x = self.adaptavgpool2d(x)
        B, C = x.shape[:2]  # x shape: (BXCXencode_dimXencoder_dimXencoder_dim)
        x = x.permute(0, 2, 3, 1).reshape(B, -1, C)  # x shape (BXpixel_dimXC), where pixel_dim=encoder_dim*encoder_dim
        return x


class EncoderTransformer(nn.Module):  # ViT
    def __init__(self, model_name):
        super(EncoderTransformer).__init__()

    def forward(self, x):
        pass
