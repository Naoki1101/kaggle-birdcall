import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cel = nn.CrossEntropyLoss()

    def forward(self, yhat, y):
        return self.cel(yhat, y)