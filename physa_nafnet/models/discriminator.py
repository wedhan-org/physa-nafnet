import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=64, n_layers=3):
        super().__init__()
        kw = 4
        padw = 1
        seq = [spectral_norm(nn.Conv2d(input_nc, ndf, kw, 2, padw)), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            seq += [
                spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kw, 2, padw, bias=False)),
                nn.InstanceNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        seq += [spectral_norm(nn.Conv2d(ndf * nf_mult, 1, kw, 1, padw))]
        self.model = nn.Sequential(*seq)

    def forward(self, input):
        return self.model(input)


class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, num_D=3):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers)
            setattr(self, 'layer' + str(i), netD.model)
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, input):
        result = []
        down = input
        for i in range(self.num_D):
            model = getattr(self, 'layer' + str(i))
            result.append(model(down))
            if i != self.num_D - 1:
                down = self.downsample(down)
        return result