import numpy as np

def transformer_encoder_layer(X: np.ndarray, weights: dict, num_heads: int, eps: float = 1e-5) -> np.ndarray:
    """
    Forward pass of a single Transformer Encoder Layer.

    Args:
        X: Input tensor of shape (batch_size, seq_len, d_model)
        weights: Dictionary containing all weight matrices and normalization parameters
        num_heads: Number of attention heads
        eps: Epsilon for layer normalization

    Returns:
        Output tensor of shape (batch_size, seq_len, d_model)
    """
    # Your code here
    def layernom(x,g,b):
        mean=np.mean(x,axis=-1,keepdims=True)
        var=np.var(x,axis=-1,keepdims=True)
        x_=(x-mean)/np.sqrt(var+eps)
        return g*x_+b 
    
    def softmax(x):
        mx=np.max(x,axis=-1,keepdims=True)
        return np.exp(x-mx)/np.sum(np.exp(x-mx),axis=-1,keepdims=True)

    def mhattn(x,weights):
        W_q=weights["W_q"]
        W_k=weights["W_k"]
        W_v=weights["W_v"]
        W_o=weights["W_o"]

        b,seq_len,d_model=x.shape
        d_head=d_model//num_heads

        Q=x @ W_q
        K=x @ W_k
        V=x @ W_v

        Q=Q.reshape(b,seq_len,num_heads,d_head)
        K=K.reshape(b,seq_len,num_heads,d_head)
        V=V.reshape(b,seq_len,num_heads,d_head)

        Q=Q.swapaxes(1,2)
        K=K.swapaxes(1,2)
        V=V.swapaxes(1,2)

        attn_scores= (Q @ K.swapaxes(2,3)) / np.sqrt(d_head)

        attn_weights=softmax(attn_scores)

        res= attn_weights @ V
        res= res.swapaxes(1,2)
        res=res.reshape(b,seq_len,d_model)
        out=res @ W_o

        return out 
    
    def ffn(x,w1,b1,w2,b2):
        z1=x @ w1 + b1
        a1=np.maximum(z1,0)
        return a1 @ w2 + b2 
    
    x1=mhattn(X,weights)
    x2=layernom(X+x1,weights["gamma1"],weights["beta1"])
    x3=ffn(x2,weights["W1"],weights["b1"],weights["W2"],weights["b2"])
    res=layernom(x3+x2,weights["gamma2"],weights["beta2"])
    return res





