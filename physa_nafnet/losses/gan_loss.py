import torch
import torch.nn as nn

class MultiScaleGANLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.loss_fn = nn.MSELoss().to(device)

    def forward(self, input_list, target_is_real):
        loss = 0
        for pred in input_list:
            if target_is_real:
                target = torch.ones_like(pred)
            else:
                target = torch.zeros_like(pred)
            loss += self.loss_fn(pred, target)
        return loss / len(input_list)