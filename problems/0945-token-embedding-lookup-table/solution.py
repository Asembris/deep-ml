import numpy as np

def token_embedding_lookup(vocab_size: int, embed_dim: int, token_ids: list, seed: int = 0) -> list:
    """
    Build a random embedding table of shape (vocab_size, embed_dim) using
    np.random.default_rng(seed).standard_normal(...), then return the rows
    corresponding to token_ids as a nested list (rounded to 4 decimals).
    """
    rng=np.random.default_rng(seed)
    emb=rng.standard_normal((vocab_size,embed_dim))
    token_ids=np.array(token_ids,dtype=int)
    res=emb[token_ids]
    return np.round(res,4).tolist()
