import torch
import torch.nn as nn
import torch.nn.functional as F

from .naf_blocks import NAFBlock
from .attention import MultiHeadSelfAttention, CoordAtt
from .bloch_layer import DifferentiableBlochLayer


class PhySAGenerator(nn.Module):
    def __init__(self, img_channel=2, width=32, enc_blk_nums=[1, 1, 1, 28], middle_blk_num=1, dec_blk_nums=[1, 1, 1, 1],
                 dropout_rate=0.1, bloch_params=None):
        super().__init__()
        self.intro = nn.Conv2d(img_channel, width, 3, 1, 1)
        self.ending = nn.Conv2d(width, 5, 3, 1, 1)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.ca_encoders = nn.ModuleList()
        self.ca_decoders = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan, drop_out_rate=dropout_rate) for _ in range(num)]))
            self.ca_encoders.append(CoordAtt(chan, chan))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan *= 2

        self.middle_blks = nn.Sequential(*[NAFBlock(chan, drop_out_rate=dropout_rate) for _ in range(middle_blk_num)])
        self.middle_attn = MultiHeadSelfAttention(chan, num_heads=8)

        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)))
            chan //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan, drop_out_rate=dropout_rate) for _ in range(num)]))
            self.ca_decoders.append(CoordAtt(chan, chan))

        if bloch_params is None:
            bloch_params = {}
        self.bloch = DifferentiableBlochLayer(**bloch_params)

    def forward(self, x):
        x = self.intro(x)
        encs = []
        for enc, ca, down in zip(self.encoders, self.ca_encoders, self.downs):
            x = enc(x)
            x = ca(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)
        x = x + self.middle_attn(x)

        for dec, ca, up, skip in zip(self.decoders, self.ca_decoders, self.ups, encs[::-1]):
            x = up(x)
            x += skip
            x = dec(x)
            x = ca(x)

        out = self.ending(x)

        pd = torch.sigmoid(out[:, 0:1])
        t1 = torch.sigmoid(out[:, 1:2])
        t2 = torch.sigmoid(out[:, 2:3])
        water = F.relu(out[:, 3:4])
        fat = F.relu(out[:, 4:5])

        r_t1, r_t2, stir = self.bloch(pd, t1, t2)

        return {
            "virtual_stir": stir,
            "virtual_water": water,
            "fat": fat,
            "reconstructed_t1": r_t1,
            "reconstructed_t2": r_t2
        }


class ReconstructionGenerator(nn.Module):
    def __init__(self, img_channel=2, out_channel=2, width=32, enc_blk_nums=[1, 1, 1, 8], dropout_rate=0.1):
        super().__init__()
        self.intro = nn.Conv2d(img_channel, width, 3, 1, 1)
        self.ending = nn.Conv2d(width, out_channel, 3, 1, 1)
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan, drop_out_rate=dropout_rate) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan *= 2

        self.middle_blks = nn.Sequential(*[NAFBlock(chan, drop_out_rate=dropout_rate) for _ in range(1)])

        for num in enc_blk_nums[::-1]:
            self.ups.append(nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)))
            chan //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan, drop_out_rate=dropout_rate) for _ in range(num)]))

    def forward(self, x):
        x = self.intro(x)
        encs = []
        for enc, down in zip(self.encoders, self.downs):
            x = enc(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for dec, up, skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x += skip
            x = dec(x)

        return self.ending(x)