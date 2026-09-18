import numpy as np

def multiquery_attention(X: np.ndarray, W_queries: list, W_key: np.ndarray, W_value: np.ndarray, W_out: np.ndarray) -> np.ndarray:
    """
    Compute Multi-Query Attention.
    
    Args:
        X: Input array of shape (seq_len, d_model)
        W_queries: List of query weight matrices, each (d_model, d_k), one per head
        W_key: Shared key weight matrix of shape (d_model, d_k)
        W_value: Shared value weight matrix of shape (d_model, d_v)
        W_out: Output projection matrix of shape (num_heads * d_v, d_model)
    
    Returns:
        Output array of shape (seq_len, d_model), rounded to 4 decimal places
    """

    d_model,d_k=W_key.shape 
    num_heads=len(W_queries)
    d_v = W_value.shape[-1]
    seq_len=X.shape[0]
    res=np.zeros((seq_len,num_heads * d_v))

    K= X @ W_key
    V= X @ W_value

    def comp_attn(q,k,v):

        scores= (q @ k.T)/np.sqrt(d_k)
        mx=np.max(scores,axis=-1,keepdims=True)
        attn=np.exp(scores-mx)/np.sum(np.exp(scores-mx),axis=-1,keepdims=True)
        return attn @ v

    for i in range(num_heads):
        W_q=np.array(W_queries[i])
        Q_h= X @ W_q

        attn=comp_attn(Q_h,K,V)
        res[:,i*d_k:(i+1)*d_k]=attn

    return res @ W_out





