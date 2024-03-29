import pandas as pd
import numpy as np

import torch
from torch.utils import data

from typing import Literal

EncTypes = Literal["palmtree", "vocab"]


class EnergyPredictionDataset(data.Dataset):
    def __init__(
        self, data_df: pd.DataFrame, enc_type: EncTypes = "palmtree", mean: bool = False
    ):
        super().__init__()

        if enc_type == "palmtree":
            if mean:
                x = data_df.bb_embeddings.apply(lambda x: np.mean(x, axis=0)).values
            else:
                x = data_df.bb_embeddings.values
        else:
            x = np.array(data_df.encoded_bb.tolist(), dtype=object)

        y = data_df.energy.values
        self.x_train = list(map(lambda x: torch.tensor(x), x))
        self.y_train = list(map(lambda y: torch.tensor(y), y))

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx: int):
        return self.x_train[idx], self.y_train[idx]
