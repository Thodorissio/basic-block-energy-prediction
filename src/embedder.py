import sys
import numpy as np
from typing import List
import os

palmtree_path = "C:/Users/thodo/Documents/sxoli/diplomatiki/PalmTree/pre-trained-model/pre_trained_model"

from pre_trained_palmtree import eval_utils as utils


def encode(text: List[str]) -> np.ndarray:

    palmtree = utils.UsableTransformer(
        model_path=f"{palmtree_path}/palmtree/transformer.ep19",
        vocab_path=f"{palmtree_path}/palmtree/vocab",
    )
    embeddings = palmtree.encode(text)

    return embeddings
