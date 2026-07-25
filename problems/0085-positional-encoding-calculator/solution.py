import numpy as np

def pos_encoding(position: int, d_model: int):
    pos = np.arange(position)[:, None]; i = np.arange(d_model)[None, :]; pos_encoding = np.where(i % 2 == 0, np.sin(pos / (10000 ** (i / d_model))), np.cos(pos / (10000 ** ((i - 1) / d_model))))
    pos_encoding = np.float16(pos_encoding)
    return pos_encoding