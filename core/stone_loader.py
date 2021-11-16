from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import zoom
import torch
from torch.utils.data import Dataset

import tifffile as tiff

from core.transforms import get_transforms


class Stone(Dataset):
    def __init__(self, images_dir: Path, label_xlx: Path, transformers):
        self._transformers = transformers
        self._ds_root = images_dir
        self._label_root = label_xlx
        self._lb = np.squeeze(pd.read_excel(str(self._label_root)).to_numpy(), axis=-1)
        f_list = list(self._ds_root.glob("*.tif"))
        f_list.sort(key=lambda p: int(p.stem.split("-")[0]))
        self._f_list = f_list

        assert len(self._f_list) == len(self._lb)

    def __len__(self):
        return len(self._f_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        n_d_im = tiff.imread(str(self._f_list[idx]))
        n_d_im = zoom(n_d_im, (0.5, 0.5, 0.5))
        lb = torch.as_tensor(self._lb[idx])
        if self._transformers is not None:
            n_d_im = self._transformers(n_d_im)

        return n_d_im, lb


# if __name__ == '__main__':
#     ds = Stone(
#         images_dir=Path("/home/lizard/Documents/Code/Project/stoneRegression/data/TheCNNmodel/Res-01"),
#         label_xlx=Path("/home/lizard/Documents/Code/Project/stoneRegression/data/TheCNNmodel/Labels.xlsx"),
#         transformers=get_transforms(0.5, 0.5, 10., n_channel=150)
#     )
#
#     print(ds[1][0].shape)
