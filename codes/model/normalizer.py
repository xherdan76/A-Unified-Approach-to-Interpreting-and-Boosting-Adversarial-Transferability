import torch
import torch.nn as nn


class Normalize(nn.Module):

    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.tensor(mean).float()
        self.std = torch.tensor(std).float()

    def forward(self, x):
        return (x - self.mean.to(x.device)[None, :, None, None]
                ) / self.std.to(x.device)[None, :, None, None]
