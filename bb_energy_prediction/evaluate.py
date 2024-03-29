import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from typing import Union, Optional
from .models import LSTM_Regressor, Simple_Regressor

from .data_utils import remove_addresses, encode_bb_from_vocab
from .embedder import encode


def evaluate(
    model: Union[LSTM_Regressor, Simple_Regressor],
    val_loader: DataLoader,
) -> dict[str, np.ndarray]:
    """evaluate model on val data

    Args:
        model (Union): model
        val_loader (DataLoader): val_loader

    Returns:
        dict[str, np.ndarray]: preds, true energies
    """

    preds = []
    true_energies = []

    for features, labels in val_loader:
        features, labels = features.cuda(), labels.cuda()
        output = model(features)

        preds.append(output.tolist())
        true_energies.append(labels.tolist())

    preds = np.array(preds).flatten()
    true_energies = np.array(true_energies).flatten()

    preds_dict = {"preds": preds, "true_energies": true_energies}

    return preds_dict


def predict(
    model: Union[LSTM_Regressor, Simple_Regressor],
    test_bbs: list,
    vocab: Optional[dict] = None,
) -> np.ndarray:
    """get test bbs energy predictions given a model

    Args:
        model (Union[LSTM_Regressor, Simple_Regressor]): model
        test_bbs (Union[list, list[list]]): test bbs or test bbs embeddings in the palmtree scenario
        vocab (Optional[dict], optional): vocab used for custom embs model. Defaults to None.

    Returns:
        np.ndarray: energy predictions
    """

    if isinstance(model, LSTM_Regressor) and model.custom:
        processed_bbs = [remove_addresses(bb) for bb in test_bbs]
        if vocab is None:
            raise ValueError(
                "When using model with custom embeddings you should also provide not None vocab."
            )
        encoded_bbs = [encode_bb_from_vocab(bb, vocab) for bb in processed_bbs]
    else:
        encoded_bbs = test_bbs.copy()

    batch_size = 256

    if isinstance(model, LSTM_Regressor):
        preds = torch.tensor([]).cuda()
        model.eval()
        with torch.no_grad():
            for i in range(0, len(encoded_bbs), batch_size):
                batch_bbs = encoded_bbs[i : (i + batch_size)]
                batch_bbs = list(map(lambda x: torch.tensor(x).cuda(), batch_bbs))
                padded_seq = pad_sequence(batch_bbs, batch_first=True, padding_value=0)
                batch_preds = model(padded_seq)
                preds = torch.cat([preds, batch_preds])
    else:
        encoded_bbs = np.array(
            [np.mean(bb, axis=0, dtype=np.float32) for bb in encoded_bbs]
        )
        encoded_bbs = torch.tensor(encoded_bbs).cuda()
        model.eval()
        with torch.no_grad():
            preds = model(encoded_bbs).flatten()

    preds = np.array([pred.cpu().numpy() for pred in preds])

    return preds
