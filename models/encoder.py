import torch.nn as nn
import torchvision.models as models
# import torch.nn.functional as F
from collections import OrderedDict


class EncoderCNN(nn.Module):
    def __init__(self, params):
        super(EncoderCNN, self).__init__()
        model_name = params.get('model_name', 'resnet50')
        if model_name == 'resnet50':
            resnet50_model = models.resnet50(pretrained=True)  # need to remove adaptive avg. pooling + fc layer
            self.pretrained_model = nn.Sequential(OrderedDict([
                (name, params) for name, params in list(resnet50_model.named_children())[:-2]]))
        feature_size = params.get('feature_size', 16)  # feature size = 14/16
        self.adaptavgpool2d = nn.AdaptiveAvgPool2d(feature_size)  # feature_size=14/16
        if not params.get('train_all_model', False):
            self.freeze_model_weights()
        unfreeze_params = params.get('unfreeze_params', {})
        self.unfreeze_params = unfreeze_params

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

    def unfreeze_model_weights(self, epoch):  # unfreeze parts of pretrained model starting from a specific epoch
        unfreeze_flag = False
        if self.unfreeze_params.get('tune', False):
            if self.unfreeze_params.get('epoch', 1) == epoch:
                layers_name = self.unfreeze_params.get('layers_name', [])
                if len(layers_name) > 0:
                    self.unfreeze_weights(layers_name)
                    unfreeze_flag = True
                self.unfreeze_params['tune'] = False
        return unfreeze_flag

    def forward(self, x):
        x = self.pretrained_model(x)
        x = self.adaptavgpool2d(x)
        B, C = x.shape[:2]  # x shape: (B, C, enc_dim, enc_dim)
        x = x.view(B, C, -1).permute(0, 2, 1)  # x shape (B, pixel_dim, C), where pixel_dim=enc_dim*enc_dim
        # x = x.permute(0, 2, 3, 1).reshape(B, -1, C)  # x shape (B, pixel_dim, C), where pixel_dim=enc_dim*enc_dim
        return x
