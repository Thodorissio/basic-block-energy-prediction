import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader

from typing import Union
from .models import LSTM_Regressor, Simple_Regressor


class EarlyStopper:
    def __init__(self, patience: int = 10, min_delta: float = 1e-3):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss: float):
        if validation_loss < self.min_validation_loss - self.min_delta:
            self.min_validation_loss = validation_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def train_model(
    model: Union[LSTM_Regressor, Simple_Regressor],
    train_loader: DataLoader,
    val_loader: DataLoader,
    lr: float = 1e-2,
    epochs: int = 50,
    verbose: bool = False,
    lr_decay: float = 0.6,
    lr_decay_step: int = 9,
    early_stopping: bool = True,
) -> dict:
    """function that trains model for energy prediction

    Args:
        model (Union[LSTM_Regressor, Simple_Regressor]): mode
        train_loader (DataLoader): train_loader
        val_loader (DataLoader): val_loader
        lr (float, optional): learning rate. Defaults to 1e-2.
        epochs (int, optional): epochs. Defaults to 100.
        verbose (bool, optional): print loss during training. Defaults to False.
        lr_decay (float, optional): learning rate decay every lr_decay_step. Defaults to 0.1.
        lr_decay_step (int, optional): epochs for lr decay. Defaults to 10.
        early_stopping (bool, optional): enable early stopping. Defaults to True.

    Returns:
        dict: model, train loss, val loss, epochs model trained
    """

    if early_stopping:
        early_stopper = EarlyStopper()

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=optimizer, step_size=lr_decay_step, gamma=lr_decay
    )

    model.cuda()

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        losses = []
        for features, labels in train_loader:
            features = features.cuda()
            labels = labels.cuda()

            output = model(features)

            # We use RMSE Loss
            loss = torch.sqrt(criterion(output.squeeze(), labels.float()) + 1e-8)

            optimizer.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()
            losses.append(loss.item())

        val_loss = []
        model.eval()
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.cuda(), labels.cuda()
                output = model(features)

                # We use RMSE Loss
                v_loss = torch.sqrt(criterion(output.squeeze(), labels.float()) + 1e-8)

                val_loss.append(v_loss.item())

        if verbose:
            print(
                f"Epoch: {epoch+1}/{epochs}, RMSE Train Loss: {round(np.mean(losses), 5)}, RMSE Val Loss: {round(np.mean(val_loss), 5)}"
            )

        train_losses.append(np.mean(losses))
        val_losses.append(np.mean(val_loss))

        if early_stopping:
            if early_stopper.early_stop(val_losses[-1]):
                break

        scheduler.step()

    model_training = {
        "model": model,
        "train_loss": train_losses,
        "val_loss": val_losses,
        "epochs_trained": epoch,
    }

    return model_training
