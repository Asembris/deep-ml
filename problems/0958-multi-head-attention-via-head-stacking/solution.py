import numpy as np

def multi_head_attention_stack(x, heads):
    """
    Multi-head causal self-attention by stacking single-head attention modules.

    Args:
        x: array-like of shape (B, T, d_in)
        heads: list of H tuples (W_q, W_k, W_v); each weight is shape (d_in, d_head)

    Returns:
        numpy.ndarray of shape (B, T, H * d_head)
    """
    # Your code here
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
    res=[]
    for (i,H) in enumerate (heads,start=0):
        W_q,W_k,W_v=H 
        context=attn(x,W_q,W_k,W_v)
        res.append(context)
    return np.concatenate(res, axis=-1)




