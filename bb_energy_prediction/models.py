import torch
from torch import nn

from typing import Optional


class LSTM_Regressor(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        vocab_size: Optional[int] = None,
        custom_embs: bool = False,
        hidden_size: int = 16,
        num_layers: int = 2,
        dense_size: int = 128,
        smaller_dense_size: int = 16,
        lstm_dropout: float = 0.1,
        dense_dropout: float = 0.1,
    ) -> None:
        """LSTM Model for energy prediction"""

        super().__init__()

        self.custom = custom_embs
        if self.custom:
            if vocab_size is None:
                raise ValueError("Should provide 'vocab_size' for custom embeddings")
            self.embedding = nn.Embedding(vocab_size, embedding_size)

        self.num_layers = num_layers
        self.hidden_size = hidden_size

        if self.num_layers > 1:
            lstm_dropout = lstm_dropout
        else:
            lstm_dropout = 0

        self.lstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

        self.dense = nn.Linear(hidden_size, dense_size)
        self.dropout = nn.Dropout(dense_dropout)
        self.smaller_dense_size = nn.Linear(dense_size, smaller_dense_size)
        self.regressor = nn.Linear(smaller_dense_size, 1)

    def forward(self, x):
        batch_size = x.size(0)

        if self.custom:
            x = self.embedding(x)

        hidden = (
            torch.zeros((self.num_layers, batch_size, self.hidden_size)).cuda(),
            torch.zeros((self.num_layers, batch_size, self.hidden_size)).cuda(),
        )

        lstm_out, hidden_state = self.lstm(x, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_size)

        out = self.dense(lstm_out)
        out = self.dropout(out)
        out = self.smaller_dense_size(out)
        out = self.regressor(out)

        out = out.view(batch_size, -1)
        out = out[:, -1]

        return out


class Simple_Regressor(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        layers: list[int],
        dropout: float = 0.15,
        regressor_dropout: float = 0.05,
    ) -> None:
        """Dense model for energy prediction, using mean of PalmTree instructions' embeddings"""
        super().__init__()

        self.layers_list = []

        for i, dense in enumerate(layers):
            if i == 0:
                self.layers_list.append(nn.Linear(embedding_size, dense).cuda())
            else:
                self.layers_list.append(nn.Linear(layers[i - 1], dense).cuda())

        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(regressor_dropout)
        self.regressor = nn.Linear(layers[-1], 1)

    def forward(self, x):
        for dense in self.layers_list:
            x = dense(x)
            if dense == self.layers_list[-1]:
                x = self.dropout2(x)
            else:
                x = self.dropout(x)

        out = self.regressor(x)

        return out
