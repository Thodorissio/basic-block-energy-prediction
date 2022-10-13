import tensorflow as tf
import numpy as np


class LSTM_Model(tf.keras.Model):
    def __init__(self, units) -> None:
        super(LSTM_Model, self).__init__()
        self.units = units

        self.lstm = tf.keras.layers.LSTM(
            self.units,
            return_sequences=True,
            return_state=True,
            dropout=0.1,
        )

    def call(self, x):

        output, state, seq = self.lstm(x)

        return output, state, seq
