import tqdm
import numpy as np
import pandas as pd

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from typing import List, Tuple, Optional

import dataset


def read_bb_data(bb_path: str, bb_energy_path: str) -> pd.DataFrame:
    """Reads bb energy data and outputs them in a dataframe"""

    bbs = {}

    # Reading basic block data
    with open(bb_path) as fIn:
        for line in tqdm.tqdm(fIn, desc="Read file"):
            # line start with "@" symbols the beggining of a new basic block
            if line[0] == "@":
                bb_name = line.split()[-1].rstrip(":")
                bbs[bb_name] = []
            else:
                bbs[bb_name].append(line.split("=")[0].rstrip())
    bbs_energy = {}

    # Reading basic block data
    with open(bb_energy_path) as fIn:
        for line in tqdm.tqdm(fIn, desc="Read file"):
            # line start with "@" symbols the begging of a new basic block
            line = line.split(":")
            bb_name = line[0].split()[-1]
            bbs_energy[bb_name] = float(line[-1].strip())

    bbs_df = pd.DataFrame({"bb_name": bbs.keys(), "bb": bbs.values()})
    bbs_energy_df = pd.DataFrame(
        {"bb_name": bbs_energy.keys(), "energy": bbs_energy.values()}
    )

    df = bbs_df.merge(bbs_energy_df, on="bb_name", how="inner")

    return df


def remove_addresses(bb: list[str]) -> list[str]:

    clean_bb = []
    for inst in bb:
        inst_list = inst.split()
        clean_inst = [tok.replace(",", "") for tok in inst_list if len(tok) < 8]
        clean_bb.append(" ".join(clean_inst))

    return clean_bb


def encode_bb_from_vocab(bb: list[str], vocab: dict, max_insts: int = 20) -> list:

    encoded_bb = []
    for inst in bb[:max_insts]:

        if inst in vocab.keys():
            encoded_bb.append(vocab[inst])
        else:
            encoded_bb.append(vocab["UNK"])

    if len(encoded_bb) < max_insts:
        encoded_bb.extend([vocab["PAD"] for _ in range(len(encoded_bb), max_insts)])

    return encoded_bb


def preprocess_bb_df(
    df: pd.DataFrame, max_instructions: int = 20, max_energy: int = 10
) -> pd.DataFrame:

    if "bb_name" in df.columns:
        clean_df = df.drop(columns=["bb_name"])

    # cut all instructions above max_instructions for each bb
    clean_df["bb"] = clean_df.bb.apply(lambda x: x[:max_instructions])
    # remove address constants
    clean_df["bb"] = clean_df.bb.map(remove_addresses)

    # Remove bbs with outlier energy
    clean_df = clean_df[clean_df.energy > 0.0]
    clean_df = clean_df[clean_df.energy <= max_energy]

    clean_df = clean_df.reset_index(drop=True)

    return clean_df


def pad_collate_fn(
    data: List[Tuple[torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor]:

    """Custom collate function that pads uneven sequenses

    Args:
        data (List[Tuple[torch.Tensor, torch.Tensor]]): unpadded data

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: padded sequences, lenghts of sequences, labels
    """

    data.sort(key=lambda x: x[0].shape[0], reverse=True)
    sequences, label = zip(*data)
    # lengths = [len(seq) for seq in sequences]
    padded_seq = pad_sequence(sequences, batch_first=True, padding_value=0)

    return (
        padded_seq,
        # torch.from_numpy(np.array(lengths)),
        torch.from_numpy(np.array(label)),
    )


def get_palmtree_data_dict(
    data_df: pd.DataFrame,
    split: float = 0.9,
    mean: bool = False,
    batch_size: int = 32,
    random_state: Optional[int] = None,
) -> dict:

    split = 0.9
    data_df = data_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    bb_df_train = data_df[: int(split * len(data_df))]
    bb_df_val = data_df[int(split * len(data_df)) :]

    train_data = dataset.EnergyPredictionDataset(bb_df_train, mean=mean)
    if mean:
        train_loader = DataLoader(
            train_data, batch_size=batch_size, shuffle=False, drop_last=True
        )
    else:
        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
            collate_fn=pad_collate_fn,
        )

    val_data = dataset.EnergyPredictionDataset(bb_df_val, mean=mean)
    if mean:
        val_loader = DataLoader(
            val_data, batch_size=batch_size, shuffle=False, drop_last=True
        )
    else:
        val_loader = DataLoader(
            val_data,
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
            collate_fn=pad_collate_fn,
        )

    data_loaders = {
        "train_loader": train_loader,
        "val_loader": val_loader,
    }

    return data_loaders


def get_vocab_data_dict(
    data_df: pd.DataFrame,
    split: float = 0.9,
    batch_size: int = 32,
    random_state: Optional[int] = None,
) -> dict:

    split = 0.9
    data_df = data_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    bb_df_train = data_df[: int(split * len(data_df))]
    bb_df_val = data_df[int(split * len(data_df)) :]

    train_data = dataset.EnergyPredictionVocabDataset(bb_df_train)
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
    )

    val_data = dataset.EnergyPredictionVocabDataset(bb_df_val)
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
    )

    data_loaders = {
        "train_loader": train_loader,
        "val_loader": val_loader,
    }

    return data_loaders
