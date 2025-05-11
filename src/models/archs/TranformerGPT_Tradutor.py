"""
@author : Gabryel-lima
@when : 2025-01-30
@homepage : https://github.com/Gabryel-lima
"""
import torch
from torch import nn


class Decoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        