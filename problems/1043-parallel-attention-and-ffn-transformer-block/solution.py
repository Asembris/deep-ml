import numpy as np

def parallel_block(x, gamma, beta, Wq, Wk, Wv, Wo, W1, b1, W2, b2):
    """
    Parallel attention + FFN transformer block.
    Returns a list of lists of shape (T, D).
    """
    x=np.array(x)
    gamma=np.array(gamma)
    beta=np.array(beta)
    Wq=np.array(Wq)
    Wk=np.array(Wk)
    Wv=np.array(Wv)
    Wo=np.array(Wo)
    W1=np.array(W1)
    W2=np.array(W2)
    b1=np.array(b1)
    b2=np.array(b2)
    def layer_norm(x):
        mean=np.mean(x,axis=-1,keepdims=True)
        var=np.var(x,axis=-1,keepdims=True)
        x_norm=(x-mean)/np.sqrt(var+1e-5)

        return x_norm * gamma + beta
    
    def attn(x):
        d_model=Wk.shape[-1]
        Q= x @ Wq 
        K= x @ Wk 
        V= x @ Wv
        attn_scores= (Q @ K.T)/np.sqrt(d_model)
        mx=np.max(attn_scores,axis=-1,keepdims=True)
        attn_weights=np.exp(attn_scores-mx)/np.sum(np.exp(attn_scores-mx),axis=-1,keepdims=True)
        res= attn_weights @ V @ Wo
        return res

    def ffn(x):
        z1= x @ W1 + b1 
        a1=np.maximum(z1, 0)
        z2= a1 @ W2 + b2 
        return z2 
    x_norm = layer_norm(x)
    res = x + attn(x_norm) + ffn(x_norm)
    return res.tolist()









