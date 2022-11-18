import torch
from torch import nn


class LSTM_Classifier(nn.Module):
    def __init__(
        self,
        emb_size: int,
        hidden_size: int = 16,
        num_layers: int = 2,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=emb_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=0.1,
        )

        self.dense = nn.Linear(hidden_size, 16)
        # It will change to regressor
        self.classifier = nn.Linear(16, 1)
        self.tanh = nn.Tanh()

    def forward(self, x, hidden):

        batch_size = x.size(0)

        lstm_out, hidden_state = self.lstm(x, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_size)

        out = self.dense(lstm_out)
        out = self.classifier(out)
        out = self.tanh(out)

        out = out.view(batch_size, -1)
        out = out[:, -1]

        return out, hidden_state

    def init_hidden(self, batch_size):

        hidden = (
            torch.zeros((self.num_layers, batch_size, self.hidden_size)).cuda(),
            torch.zeros((self.num_layers, batch_size, self.hidden_size)).cuda(),
        )

        return hidden
