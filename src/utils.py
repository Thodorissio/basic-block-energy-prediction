import numpy as np

import torch
from torch.nn.utils.rnn import pad_sequence

from typing import List, Tuple


def collate_fn(
    data: List[Tuple[torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    """Custom collate function that pads uneven sequenses

    Args:
        data (List[Tuple[torch.Tensor, torch.Tensor]]): unpadded data

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: padded sequences, lenghts of sequences, labels
    """

    data.sort(key=lambda x: x[0].shape[0], reverse=True)
    sequences, label = zip(*data)
    lengths = [len(seq) for seq in sequences]
    padded_seq = pad_sequence(sequences, batch_first=True, padding_value=0)

    return (
        padded_seq,
        torch.from_numpy(np.array(lengths)),
        torch.from_numpy(np.array(label)),
    )
