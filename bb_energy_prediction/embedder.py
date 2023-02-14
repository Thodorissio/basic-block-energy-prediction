import numpy as np
import os
from typing import List
from dotenv import load_dotenv

load_dotenv()

palmtree_path = os.getenv("PALMTREE_PATH")

from pre_trained_palmtree import eval_utils as utils


def encode(basic_block: List[str]) -> np.ndarray:
    """get palmtree embeddings for instructions consisting basic block basic block

    Args:
        basic_block (List[str]): list of assembly instructions

    Returns:
        np.ndarray: list of palmtree embeddings corresponding to basic block's instructions
    """

    palmtree = utils.UsableTransformer(
        model_path=f"{palmtree_path}/pre_trained_palmtree/palmtree/transformer.ep19",
        vocab_path=f"{palmtree_path}/pre_trained_palmtree/palmtree/vocab",
    )
    embeddings = palmtree.encode(basic_block)

    return embeddings
