import os
import argparse
import numpy as np
import torch
from pytorch_msssim import ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_absolute_error as mae
import lpips


def compute_nmse(pred, ref, eps=1e-8):
    pred = pred.astype(np.float32)
    ref = ref.astype(np.float32)
    return float(np.sum((pred - ref) ** 2) / (np.sum(ref ** 2) + eps))


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PhySA-NAFNet outputs")
    parser.add_argument('--pred_dir', type=str, required=True, help='Directory containing predictions')
    parser.add_argument('--ref_dir', type=str, required=True, help='Directory containing ground truth')
    parser.add_argument('--target', type=str, default='virtual_stir',
                        help='Target to evaluate (e.g., virtual_stir, virtual_water)')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_vgg = lpips.LPIPS(net='vgg').to(device)

    psnr_list, ssim_list, mae_list, nmse_list, lpips_list = [], [], [], [], []

    pred_files = [f for f in os.listdir(args.pred_dir) if args.target in f and f.endswith('.npy')]

    print(f"Evaluating {len(pred_files)} files for target: {args.target}")

    for f in pred_files:
        # Expected naming match, e.g., Case_000_virtual_stir.npy maps to Case_000_acquired_stir.npy
        base_name = f.replace(f"_{args.target}.npy", "")

        pred_path = os.path.join(args.pred_dir, f)
        # Using a generalized convention. Modify logic based on actual ref naming.
        ref_path = os.path.join(args.ref_dir, f"{base_name}_acquired_stir.npy")

        if not os.path.exists(ref_path):
            continue

        pred = np.load(pred_path).astype(np.float32)
        ref = np.load(ref_path).astype(np.float32)

        psnr_list.append(psnr(ref, pred, data_range=1.0))
        mae_list.append(mae(ref, pred))
        nmse_list.append(compute_nmse(pred, ref))

        # Format for torch metrics [1, 1, H, W]
        p_t = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0).to(device)
        r_t = torch.from_numpy(ref).unsqueeze(0).unsqueeze(0).to(device)

        ssim_list.append(ssim(p_t, r_t, data_range=1.0, size_average=True).item())

        # Format for LPIPS [-1, 1] RGB
        p_t_rgb = p_t.repeat(1, 3, 1, 1) * 2 - 1
        r_t_rgb = r_t.repeat(1, 3, 1, 1) * 2 - 1
        lpips_list.append(lpips_vgg(p_t_rgb, r_t_rgb).item())

    if len(psnr_list) == 0:
        print("No matching prediction-reference pairs found.")
        return

    print("=== Evaluation Results ===")
    print(f"PSNR:  {np.mean(psnr_list):.4f} ± {np.std(psnr_list):.4f}")
    print(f"SSIM:  {np.mean(ssim_list):.4f} ± {np.std(ssim_list):.4f}")
    print(f"MAE:   {np.mean(mae_list):.4f} ± {np.std(mae_list):.4f}")
    print(f"NMSE:  {np.mean(nmse_list):.4f} ± {np.std(nmse_list):.4f}")
    print(f"LPIPS: {np.mean(lpips_list):.4f} ± {np.std(lpips_list):.4f}")


if __name__ == '__main__':
    main()