import sys
import numpy as np
from typing import List
import os

palmtree_path = "C:/Users/thodo/Documents/σχολη/diplomatiki/PalmTree/pre-trained_model"

sys.path.append(palmtree_path)
import eval_utils as utils


def encode(text: List[str]) -> np.ndarray:

    palmtree = utils.UsableTransformer(model_path=f"{palmtree_path}/palmtree/transformer.ep19", vocab_path=f"{palmtree_path}/palmtree/vocab")
    embeddings = palmtree.encode(text)

    return embeddings