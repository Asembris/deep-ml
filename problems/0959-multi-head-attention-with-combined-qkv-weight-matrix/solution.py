import numpy as np

def mha_combined_qkv(x, W_qkv, W_out, num_heads):
    """
    Multi-head causal self-attention using a single combined QKV weight matrix.

    Args:
        x: array of shape (batch, num_tokens, d_in)
        W_qkv: array of shape (d_in, 3 * d_out)
        W_out: array of shape (d_out, d_out)
        num_heads: int, number of attention heads (d_out % num_heads == 0)

    Returns:
        Nested list of shape (batch, num_tokens, d_out).
    """
    def softmax(x):
        mx=np.max(x,axis=-1,keepdims=True)
        x-=mx
        res=np.exp(x)/np.sum(np.exp(x),axis=-1,keepdims=True)
        return res 

    def attn(x,w_q,w_k,w_v):
        q=x @ w_q
        k=x @ w_k
        v=x @ w_v 
        d_head=k.shape[-1]
        T=x.shape[1]
        mask=np.triu(np.ones((T, T)), k=1)

        attn_scores= (q @ k.transpose(0,2,1)) / np.sqrt(d_head)
        attn_scores=np.where(mask == 1, -np.inf,attn_scores)
        attn_weights=softmax(attn_scores)
        context=attn_weights @ v
        return context 

    x=np.array(x)
    d_out=W_qkv.shape[-1]//3
    d_head=d_out//num_heads
    Q=W_qkv[:,0:d_out]
    K=W_qkv[:,d_out:2*d_out]
    V=W_qkv[:,2*d_out:3*d_out]
    res=[]
    for i in range(num_heads):
        w_q=Q[:,i*d_head:(i+1)*d_head]
        w_k=K[:,i*d_head:(i+1)*d_head]
        w_v=V[:,i*d_head:(i+1)*d_head]
        context=attn(x,w_q,w_k,w_v)
        res.append(context)
    res=np.concatenate(res, axis=-1)
    return (res @ W_out).tolist()


