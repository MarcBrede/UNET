import torch
from torch import nn

# weighted cross entropy
class WeightedBCE(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = weights
    
    def forward(self, y_pred, y_label):
        background_loss = ((1 - y_label) * torch.log(1 - y_pred))# * self.weights[0]
        foreground_loss = ((y_label * torch.log(y_pred)))# * self.weights[1:][None, :, None, None]
        return -(background_loss.mean() + foreground_loss.mean())