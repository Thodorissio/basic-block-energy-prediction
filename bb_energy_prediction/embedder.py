import numpy as np
import os
from typing import List
from dotenv import load_dotenv

load_dotenv()

palmtree_path = os.getenv("PALMTREE_PATH")

from pre_trained_palmtree import eval_utils as utils


def encode(text: List[str]) -> np.ndarray:

    palmtree = utils.UsableTransformer(
        model_path=f"{palmtree_path}/pre_trained_palmtree/palmtree/transformer.ep19",
        vocab_path=f"{palmtree_path}/pre_trained_palmtree/palmtree/vocab",
    )
    embeddings = palmtree.encode(text)

    return embeddings
