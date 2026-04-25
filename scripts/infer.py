import os
import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt

from physa_nafnet.models.generator import PhySAGenerator


def parse_args():
    parser = argparse.ArgumentParser(description="Inference for PhySA-NAFNet")
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to best_G_AB.pth')
    parser.add_argument('--input_npy', type=str, required=True, help='Path to input NPY file')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save outputs')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    model_cfg = config['model']
    G_AB = PhySAGenerator(
        img_channel=2, width=model_cfg['width'],
        enc_blk_nums=model_cfg['encoder_blocks'],
        dec_blk_nums=model_cfg['decoder_blocks'],
        dropout_rate=model_cfg.get('dropout_rate', 0.1),
        bloch_params=config.get('bloch', {})
    ).to(device)

    G_AB.load_state_dict(torch.load(args.checkpoint, map_location=device))
    G_AB.eval()

    data = np.load(args.input_npy, allow_pickle=True).item()
    input_arr = data['input'].astype(np.float32)

    if input_arr.shape[1] != config['data']['image_size'] or input_arr.shape[2] != config['data']['image_size']:
        raise ValueError(
            f"Expected input size {config['data']['image_size']}x{config['data']['image_size']}, "
            f"but got {input_arr.shape[1:]}"
        )

    input_tensor = torch.from_numpy(input_arr).unsqueeze(0).to(device)

    with torch.no_grad():
        out_dict = G_AB(input_tensor)

    v_stir = out_dict['virtual_stir'].cpu().numpy()[0, 0]
    v_water = out_dict['virtual_water'].cpu().numpy()[0, 0]
    fat = out_dict['fat'].cpu().numpy()[0, 0]

    np.save(os.path.join(args.output_dir, "virtual_stir.npy"), v_stir)
    np.save(os.path.join(args.output_dir, "virtual_water.npy"), v_water)
    np.save(os.path.join(args.output_dir, "fat.npy"), fat)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(v_stir, cmap='gray');
    axes[0].set_title('Virtual STIR')
    axes[1].imshow(v_water, cmap='gray');
    axes[1].set_title('Virtual Water')
    axes[2].imshow(fat, cmap='gray');
    axes[2].set_title('Fat Component')

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "preview.png"), dpi=300)
    plt.close(fig)

    print(f"Inference complete. Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()