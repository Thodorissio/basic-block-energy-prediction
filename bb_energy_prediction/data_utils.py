import tqdm
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from collections import Counter
from typing import List, Tuple, Optional, Literal

from . import dataset, embedder

EncTypes = Literal["palmtree", "vocab"]


def create_data_df(data_path: str, no_duplicates: bool = False) -> pd.DataFrame:
    """Create data based on custom dataset and save it in data path"""

    energy_data_dir = os.getenv("ENERGY_DATASET_PATH")
    result_files = [f for f in os.listdir(energy_data_dir) if f.endswith("results")]
    data_df = pd.DataFrame()

    for file in result_files:
        file_df = read_bb_data(
            f"{energy_data_dir}/{file}/breaker_code.txt",
            f"{energy_data_dir}/{file}/breaker_final_energy.txt",
        )

        file_df["program_name"] = file.rsplit("_results", 1)[0]
        data_df = pd.concat([data_df, file_df], ignore_index=True)

    if no_duplicates:
        max_instructions = 25
        data_df = (
            data_df.groupby(data_df.bb.map(tuple))["energy"].median().reset_index()
        )
        data_df.bb = data_df.bb.map(list)
    else:
        max_instructions = 20
    data_df = preprocess_bb_df(data_df, max_instructions=max_instructions)

    data_df["bb_embeddings"] = data_df.bb.apply(lambda x: embedder.encode(x))
    data_df.to_pickle(data_path)

    return data_df


def get_data_df(data_path: str) -> pd.DataFrame:
    """Get dataframe containing: bb, labels, bb's program name, bb embedding"""

    data_dir = data_path.rsplit("/", 1)[0]
    data_file = data_path.split("/")[-1]

    if not os.path.exists(data_dir):
        print("Given data directory does not exist. Creating new directory.")
        os.makedirs(data_dir)

    if data_file in os.listdir(data_dir):
        data_df = pd.read_pickle(data_path)
    else:
        print("Data file does not exist. Creating new data file.")
        data_df = create_data_df(data_path)

    return data_df


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


def get_inst_vocab(data_df: pd.DataFrame) -> dict:
    """Creates instruction vocabulary based on bbs of data_df"""

    counts = Counter(inst for bb in data_df.bb.tolist() for inst in set(bb))

    vocab = {inst: i for i, (inst, _) in enumerate(counts.most_common(20000), start=2)}
    vocab["PAD"] = 0
    vocab["UNK"] = 1

    return vocab


def encode_bb_from_vocab(bb: list[str], vocab: dict, max_insts: int = 20) -> list:
    encoded_bb = []
    for inst in bb[:max_insts]:
        if inst in vocab.keys():
            encoded_bb.append(vocab[inst])
        else:
            encoded_bb.append(vocab["UNK"])

    return encoded_bb


def preprocess_bb_df(
    df: pd.DataFrame, max_instructions: int = 20, max_energy: int = 10
) -> pd.DataFrame:
    if "bb_name" in df.columns:
        clean_df = df.drop(columns=["bb_name"])
    else:
        clean_df = df.copy()

    # remove empty basic blocks
    clean_df = clean_df[clean_df.bb.map(len) > 0]
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
    padded_seq = pad_sequence(sequences, batch_first=True, padding_value=0)

    return (
        padded_seq,
        torch.from_numpy(np.array(label)),
    )


def get_data_dict(
    data_df: pd.DataFrame,
    enc_type: EncTypes,
    split: float = 0.9,
    mean: bool = False,
    batch_size: int = 32,
    random_state: Optional[int] = None,
) -> dict:
    split = 0.9
    data_df = data_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    bb_df_train = data_df[: int(split * len(data_df))]
    bb_df_val = data_df[int(split * len(data_df)) :]

    train_data = dataset.EnergyPredictionDataset(
        bb_df_train, enc_type=enc_type, mean=mean
    )
    val_data = dataset.EnergyPredictionDataset(bb_df_val, enc_type=enc_type, mean=mean)
    if mean and enc_type == "palmtree":
        train_loader = DataLoader(
            train_data, batch_size=batch_size, shuffle=False, drop_last=True
        )
        val_loader = DataLoader(
            val_data, batch_size=batch_size, shuffle=False, drop_last=True
        )
    else:
        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
            collate_fn=pad_collate_fn,
        )
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
