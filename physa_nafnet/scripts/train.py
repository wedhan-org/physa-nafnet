import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import itertools
from torch.utils.data import DataLoader
from tqdm import tqdm
from pytorch_msssim import SSIM

from physa_nafnet.data.dataset import VFS_Npy_Dataset
from physa_nafnet.models.generator import PhySAGenerator, ReconstructionGenerator
from physa_nafnet.models.discriminator import MultiscaleDiscriminator
from physa_nafnet.losses.gan_loss import MultiScaleGANLoss
from physa_nafnet.losses.physics_losses import dual_guide_loss, ExclusionLoss


def parse_args():
    parser = argparse.ArgumentParser(description="Train PhySA-NAFNet")
    parser.add_argument('--config', type=str, required=True, help='Path to training config YAML')
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    save_dir = config['output']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Starting PhySA-NAFNet training")

    model_cfg = config['model']
    bloch_cfg = config.get('bloch', {})
    G_AB = PhySAGenerator(
        img_channel=2,
        width=model_cfg['width'],
        enc_blk_nums=model_cfg['encoder_blocks'],
        dec_blk_nums=model_cfg['decoder_blocks'],
        dropout_rate=model_cfg.get('dropout_rate', 0.1),
        bloch_params=bloch_cfg
    ).to(device)

    G_BA = ReconstructionGenerator(
        img_channel=2,
        out_channel=2,
        width=model_cfg['width'],
        dropout_rate=model_cfg.get('dropout_rate', 0.1)
    ).to(device)

    D_B = MultiscaleDiscriminator(input_nc=2, num_D=3).to(device)
    D_A = MultiscaleDiscriminator(input_nc=2, num_D=3).to(device)

    train_cfg = config['training']
    opt_G = optim.AdamW(itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=train_cfg['learning_rate'],
                        betas=(0.5, 0.999))
    opt_D = optim.AdamW(itertools.chain(D_B.parameters(), D_A.parameters()), lr=train_cfg['learning_rate'],
                        betas=(0.5, 0.999))

    data_cfg = config['data']
    train_ds = VFS_Npy_Dataset(data_cfg['root'], data_cfg['split_csv'], mode='train', img_size=data_cfg['image_size'])
    train_loader = DataLoader(train_ds, batch_size=train_cfg['batch_size'], shuffle=True,
                              num_workers=train_cfg['num_workers'])

    val_ds = VFS_Npy_Dataset(data_cfg['root'], data_cfg['split_csv'], mode='val', img_size=data_cfg['image_size'])
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=1)

    criterionGAN = MultiScaleGANLoss(device)
    criterionCycle = nn.L1Loss().to(device)
    excl_loss = ExclusionLoss().to(device)
    l1 = nn.L1Loss().to(device)
    ssim_metric = SSIM(data_range=1.0, size_average=True, channel=1).to(device)

    w_cfg = config['loss_weights']
    best_ssim = 0.0

    for epoch in range(train_cfg['epochs']):
        G_AB.train();
        G_BA.train();
        D_B.train();
        D_A.train()
        loop = tqdm(train_loader, desc=f"Ep {epoch + 1}")

        for batch in loop:
            real_A = batch['input'].to(device)
            real_B = batch['target_stir'].to(device)

            real_B_shuffled = real_B[torch.randperm(real_B.size(0))]
            real_B_shuffled_paired = torch.cat([real_B_shuffled, real_B_shuffled], 1)

            # --- Train Generators ---
            opt_G.zero_grad()
            out_dict = G_AB(real_A)
            fake_stir = out_dict['virtual_stir']
            water = out_dict['virtual_water']
            fat = out_dict['fat']

            fake_B_paired = torch.cat([fake_stir, water], 1)
            rec_A = G_BA(fake_B_paired)
            fake_A = G_BA(real_B_shuffled_paired)
            rec_stir = G_AB(fake_A)['virtual_stir']

            loss_G_GAN = (criterionGAN(D_B(fake_B_paired), True) + criterionGAN(D_A(fake_A), True)) * w_cfg['gan']
            loss_cycle = (criterionCycle(rec_A, real_A) + criterionCycle(rec_stir, real_B_shuffled)) * w_cfg['cycle']
            loss_guide = dual_guide_loss(water, fat, real_A[:, 1:2], real_B) * w_cfg['dual_guide']
            loss_excl = excl_loss(water, fat) * w_cfg['exclusion']
            loss_bloch = (l1(out_dict['reconstructed_t1'], real_A[:, 0:1]) + l1(out_dict['reconstructed_t2'],
                                                                                real_A[:, 1:2])) * w_cfg['bloch']

            loss_G = loss_G_GAN + loss_cycle + loss_guide + loss_excl + loss_bloch
            loss_G.backward()
            opt_G.step()

            # --- Train Discriminators ---
            opt_D.zero_grad()
            loss_D_B = (criterionGAN(D_B(real_B_shuffled_paired), True) + criterionGAN(D_B(fake_B_paired.detach()),
                                                                                       False)) * 0.5
            loss_D_A = (criterionGAN(D_A(real_A), True) + criterionGAN(D_A(fake_A.detach()), False)) * 0.5
            loss_D = loss_D_B + loss_D_A
            loss_D.backward()
            opt_D.step()

            loop.set_postfix({'G_loss': f"{loss_G.item():.4f}"})

        # --- Validation ---
        G_AB.eval()
        val_ssim_total = 0
        with torch.no_grad():
            for batch in val_loader:
                real_A = batch['input'].to(device)
                real_B = batch['target_stir'].to(device)
                fake_stir = G_AB(real_A)['virtual_stir']
                val_ssim_total += ssim_metric(fake_stir, real_B).item()

        avg_ssim = val_ssim_total / len(val_loader)
        print(f"Validation SSIM: {avg_ssim:.4f}")

        if avg_ssim > best_ssim:
            best_ssim = avg_ssim
            torch.save(G_AB.state_dict(), os.path.join(save_dir, 'best_G_AB.pth'))
            print("New best model saved.")

        torch.save(G_AB.state_dict(), os.path.join(save_dir, 'latest_G_AB.pth'))


if __name__ == '__main__':
    main()