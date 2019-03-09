import torch

from torch import nn


class Model(nn.Module):
    def __init__(self):
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, *input):
        pass


class Encoder(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass


class Decoder(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass