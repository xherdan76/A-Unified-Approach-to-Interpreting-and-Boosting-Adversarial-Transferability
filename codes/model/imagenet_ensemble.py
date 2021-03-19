# import codes.basic_functions.ourpretrainedmodels as pretrainedmodels
import pretrainedmodels
import torch
import torch.nn as nn


class ImagenetEnsemble(nn.Module):

    def __init__(self, ):
        super(ImagenetEnsemble, self).__init__()

        self.archs = ['resnet34', 'resnet152', 'densenet121']

        for model_type in self.archs:
            model = pretrainedmodels.__dict__[model_type](
                num_classes=1000, pretrained='imagenet').eval()
            for param in model.parameters():
                param.requires_grad = False
            setattr(self, model_type, model)

            self.input_size = model.input_size
            self.mean = model.mean
            self.std = model.std

    def forward(self, x):
        logits = 0
        for arch in self.archs:
            model = getattr(self, arch)
            logits += model(x)
        return logits / len(self.archs)
