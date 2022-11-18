import pandas as pd

import torch
from torch.utils import data


class IMDB_Dataset(data.Dataset):
    def __init__(self, data_df: pd.DataFrame):
        super().__init__()
        x = data_df.sentence_emb.values
        y = data_df.energy.values
        self.x_train = list(map(lambda x: torch.tensor(x), x))
        self.y_train = list(map(lambda y: torch.tensor(y), y))

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx: int):
        return self.x_train[idx], self.y_train[idx]
