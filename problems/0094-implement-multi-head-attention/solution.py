import numpy as np
from typing import Tuple

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Query, Key, and Value matrices.
    
    Args:
        X: Input matrix of shape (seq_len, d_model)
        W_q, W_k, W_v: Weight matrices of shape (d_model, d_model)
    
    Returns:
        Q, K, V matrices each of shape (seq_len, d_model)
    """
    Q=np.dot(X,W_q)
    K=np.dot(X,W_k)
    V=np.dot(X,W_v)
    return Q,K,V
    pass

def self_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Compute scaled dot-product self-attention.
    
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_k)
    
    Returns:
        Attention output of shape (seq_len, d_k)
    """
    att_scores=np.dot(Q,K.T)/np.sqrt(K.shape[1])
    att_scores-=np.max(att_scores, axis=1, keepdims=True)
    att_weights=np.exp(att_scores)/(np.sum(np.exp(att_scores),axis=1,keepdims=True))
    res=np.dot(att_weights,V)
    return res
    pass

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, n_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    
    Args:
        Q, K, V: Matrices of shape (seq_len, d_model)
        n_heads: Number of attention heads
    
    Returns:
        Attention output of shape (seq_len, d_model)
    """
    d_model=K.shape[1]
    seq_len=K.shape[0]
    d=d_model//n_heads
    Qi=np.split(Q,n_heads,axis=1)
    Ki=np.split(K,n_heads,axis=1)
    Vi=np.split(V,n_heads,axis=1)
    res=np.zeros((K.shape[0], K.shape[1]))
    for i in range(n_heads):
        At=self_attention(Qi[i],Ki[i],Vi[i])
        res[:,i*d:(i+1)*d]=At
    return res
    pass



    