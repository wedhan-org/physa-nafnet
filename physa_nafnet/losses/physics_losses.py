import torch
import torch.nn as nn
import torch.nn.functional as F


class ExclusionLoss(nn.Module):
    """Encourages structural exclusion between separated water and fat maps."""

    def forward(self, m1, m2):
        return torch.mean(torch.sigmoid(m1) * torch.sigmoid(m2))


def dual_guide_loss(water, fat, t2, stir):
    """
    Core dual-guide strategy:
    1. Virtual water is weakly guided by acquired STIR as an indirect water-dominant comparator.
    2. Pseudo-fat guidance: fat-related signal is approximated as T2 - STIR.
    """
    water_loss = F.l1_loss(water, stir)

    pseudo_fat = F.relu(t2 - stir).detach()
    fat_loss = F.l1_loss(fat, pseudo_fat)

    return water_loss + fat_loss