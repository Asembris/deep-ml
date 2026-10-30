import numpy as np

def batched_causal_self_attention(X, W_query, W_key, W_value):
    """
    Batched causal self-attention.

    Args:
        X: nested list of shape (B, T, d_in)
        W_query, W_key, W_value: nested lists of shape (d_in, d_out)

    Returns:
        Nested list of shape (B, T, d_out) -- the context vectors.
    """
    # Your code here

    def softmax(x):
        mx=np.max(x,axis=-1,keepdims=True)
        x-=mx
        res=np.exp(x)/np.sum(np.exp(x),axis=-1,keepdims=True)
        return res

    X=np.array(X)
    W_key=np.array(W_key)
    W_query=np.array(W_query)
    W_value=np.array(W_value)

    B,T=X.shape[0],X.shape[1]
    Q=X @ W_query
    K=X @ W_key
    V=X @ W_value
    d=K.shape[-1]

    attn_scores= Q @ np.moveaxis(K,-1,-2) / np.sqrt(d)
    mask=np.triu(np.ones((B,T,T)), k=1)
    attn_scores=np.where(mask == 1, -np.inf,attn_scores)

    attn_weights=softmax(attn_scores)

    res=attn_weights @ V
    return res.tolist()

    
