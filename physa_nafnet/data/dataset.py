import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class VFS_Npy_Dataset(Dataset):
    def __init__(self, data_dir, csv_path, mode='train', img_size=512):
        self.data_dir = data_dir
        self.img_size = img_size
        self.files = []

        if csv_path is None or not os.path.exists(csv_path):
            raise ValueError(
                "A patient-level split CSV is required. "
                "The CSV must contain 'filename' and 'split' columns."
            )

        df = pd.read_csv(csv_path)

        if 'filename' in df.columns and 'split' in df.columns:
            valid_files = df[df['split'] == mode]['filename'].tolist()
            for f in valid_files:
                if os.path.exists(os.path.join(data_dir, f)):
                    self.files.append(f)
        else:
            raise ValueError("CSV must contain 'filename' and 'split' columns.")

        print(f"[{mode.upper()}] Loaded: {len(self.files)} slices")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        filename = self.files[index]
        path = os.path.join(self.data_dir, filename)

        try:
            loaded = np.load(path, allow_pickle=True)
            item = loaded.item() if (isinstance(loaded, np.ndarray) and loaded.ndim == 0) else loaded

            if isinstance(item, dict) and 'input' in item and 'label' in item:
                data = np.concatenate([item['input'], item['label']], axis=0)
            else:
                raise ValueError("NPY must be a dictionary with 'input' and 'label' keys.")

            data = data.astype(np.float32)

            if data.ndim == 3 and data.shape[2] == 3:
                data = data.transpose(2, 0, 1)

            if data.shape[1] != self.img_size:
                data = np.stack([cv2.resize(data[c], (self.img_size, self.img_size)) for c in range(data.shape[0])],
                                axis=0)

            input_tensor = torch.from_numpy(data[:2].copy())
            target_tensor = torch.from_numpy(data[2:].copy())

            return {
                "input": input_tensor,
                "target_stir": target_tensor,
                "filename": filename
            }

        except Exception as e:
            raise RuntimeError(f"Failed to load or process {path}: {e}")