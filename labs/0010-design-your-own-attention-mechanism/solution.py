import numpy as np

def attention(Q, K, V):
    """
    Compute attention over a sequence.
    
    Args:
        Q: Query matrix, shape (batch_size, query_len, dim)
        K: Key matrix, shape (batch_size, key_len, dim)
        V: Value matrix, shape (batch_size, key_len, dim)
    
    Returns:
        output: Attended values, shape (batch_size, query_len, dim)
    
    The attention mechanism should:
    1. Compute compatibility between queries and keys
    2. Convert to attention weights (non-negative, sum to 1)
    3. Use weights to compute weighted sum of values
    """
    batch_size, query_len, dim = Q.shape
    _, key_len, _ = K.shape
    
    # Your implementation here
    # Hint: Think about how queries "ask questions" and keys "provide answers"
    
    d_model=Q.shape[-1]
    d_head=d_model//4

    def attention_head(q,k,v):

        scores=q @ k.swapaxes(1,2)
        mx=np.max(scores,axis=-1,keepdims=True)
        attn_scores=np.exp(scores-mx)/np.sum(np.exp(scores-mx))
        attn_weights=attn_scores @ v 
        return attn_weights
    
    output=np.zeros_like(Q)
    for i in range(4):
        attn_head=attention_head(Q[:,:,i*d_head:(i+1)*d_head],K[:,:,i*d_head:(i+1)*d_head],V[:,:,i*d_head:(i+1)*d_head])

        output[:,:,i*d_head:(i+1)*d_head]=attn_head
    return output






