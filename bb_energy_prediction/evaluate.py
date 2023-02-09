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
    batch_size: int,
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

    if isinstance(model, LSTM_Regressor):
        lstm_model = True
        hidden_state = model.init_hidden(batch_size=batch_size)
    else:
        lstm_model = False

    for features, labels in val_loader:
        features, labels = features.cuda(), labels.cuda()
        if lstm_model:
            output, hidden_state = model(features, hidden_state)
        else:
            output = model(features)

        preds.append(output.tolist())
        true_energies.append(labels.tolist())

    preds = np.array(preds).flatten()
    true_energies = np.array(true_energies).flatten()

    preds_dict = {"preds": preds, "true_energies": true_energies}

    return preds_dict


def predict(
    model: Union[LSTM_Regressor, Simple_Regressor],
    test_bbs: Union[list, list[list]],
    vocab: Optional[dict] = None,
) -> np.ndarray:
    """get test bbs energy predictions given a model

    Args:
        model (Union[LSTM_Regressor, Simple_Regressor]): model
        test_bbs (Union[list, list[list]]): test bbs
        vocab (Optional[dict], optional): vocab used for custom embs model. Defaults to None.

    Returns:
        np.ndarray: energy predictions
    """

    processed_bbs = [remove_addresses(bb) for bb in test_bbs]

    if model.custom:
        if vocab is None:
            raise ValueError(
                "When using model with custom embeddings you should also provide not None vocab."
            )
        encoded_bbs = [encode_bb_from_vocab(bb, vocab) for bb in processed_bbs]
    else:
        encoded_bbs = [encode(bb) for bb in processed_bbs]

    encoded_bbs = list(map(lambda x: torch.tensor(x).cuda(), encoded_bbs))

    if isinstance(model, LSTM_Regressor):
        padded_seq = pad_sequence(encoded_bbs, batch_first=True, padding_value=0)
        hidden_state = model.init_hidden(len(test_bbs))
        preds, _ = model(padded_seq, hidden_state)
    else:
        mean_encoded_bbs = [np.mean(bb, axis=0) for bb in encoded_bbs]
        preds = model(mean_encoded_bbs)

    preds = np.array(preds.cpu())

    return preds
