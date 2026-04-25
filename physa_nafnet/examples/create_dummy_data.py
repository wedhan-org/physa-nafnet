import os
import argparse
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, default='examples/dummy_data', help='Output directory for dummy data')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of dummy samples to generate')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    records = []

    print(f"Generating {args.num_samples} dummy NPY samples in {args.out_dir}...")
    for i in range(args.num_samples):
        # Create random noise to simulate T1, T2, and STIR
        dummy_input = np.random.rand(2, 512, 512).astype(np.float32)
        dummy_label = np.random.rand(1, 512, 512).astype(np.float32)

        case_id = f"Case_{i // 3:03d}"
        filename = f"{case_id}_slice_{i % 3:02d}.npy"

        data_dict = {
            "input": dummy_input,
            "label": dummy_label,
            "case_id": case_id,
            "slice_id": i % 3
        }

        np.save(os.path.join(args.out_dir, filename), data_dict)

        # Simple split logic: first 60% train, 20% val, 20% test
        if i < int(args.num_samples * 0.6):
            split = 'train'
        elif i < int(args.num_samples * 0.8):
            split = 'val'
        else:
            split = 'test'

        records.append({'filename': filename, 'case_id': case_id, 'split': split})

    df = pd.DataFrame(records)
    csv_path = os.path.join(args.out_dir, 'dummy_split.csv')
    df.to_csv(csv_path, index=False)

    print(f"Done. Dummy CSV saved to {csv_path}")


if __name__ == "__main__":
    main()