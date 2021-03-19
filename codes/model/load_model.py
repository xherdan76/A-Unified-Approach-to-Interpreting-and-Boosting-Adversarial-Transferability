import pretrainedmodels
import torch.nn as nn

from .imagenet_ensemble import ImagenetEnsemble


def load_imagenet_model(model_type):
    if model_type == 'ensemble':
        model = ImagenetEnsemble()
    else:
        model = pretrainedmodels.__dict__[model_type](
            num_classes=1000, pretrained='imagenet').eval()
        for param in model.parameters():
            param.requires_grad = False
    return model
