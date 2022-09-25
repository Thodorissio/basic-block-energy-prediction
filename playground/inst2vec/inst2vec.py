import sys
import numpy as np
from typing import List

sys.path.append("../../../instruction2vec/core/")
import gen_gensim_model as gen
import instruction2vec as inst2vec

def encode_block(code_block: str, vector_size: int = 5) -> List[np.ndarray]:
    """turns code block to inst2vec embedding

    Args:
        code_block (str): block of code to be encoded
        vector_size (int, optional): size scale of embedding (for scale=1 embedding=9). Defaults to 5.

    Returns:
        List[np.ndarray]: list of code block's instruction embeddings
    """
    
    fd = open("../../instruction2vec/sample/asmcode_corpus", "r")
    asmcode_corpus = fd.read()

    model = gen.gen_instruction2vec_model(asmcode_corpus, vector_size, "test_model")
    encoded_block = []

    encoded_block = [inst2vec.instruction2vec(inst, model, vectorsize=vector_size) for inst in code_block.split('\n')]

    return encoded_block