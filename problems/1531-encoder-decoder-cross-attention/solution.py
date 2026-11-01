import numpy as np


def cross_attention(decoder_x, encoder_out, W_q, W_k, W_v, src_pad_mask=None):
    """
    Encoder-decoder attention: queries from decoder states, keys and values from encoder output.

    decoder_x:    (tgt_len, d_model)
    encoder_out:  (src_len, d_model)
    W_q, W_k:     (d_model, d_k)
    W_v:          (d_model, d_v)
    src_pad_mask: optional (src_len,) boolean / 0-1 array, True or 1 at padded source positions

    Returns (output, weights) with shapes (tgt_len, d_v) and (tgt_len, src_len).
    Masked source positions must have weight exactly 0.
    """
    Q=decoder_x @ W_q
    K=encoder_out @ W_k 
    V=encoder_out @ W_v 
    d_k=K.shape[-1]
    attn_scores= (Q @ K.T)/np.sqrt(d_k)
    if src_pad_mask is not None:
        attn_scores=np.where(src_pad_mask, -np.inf, attn_scores)
    mx=np.max(attn_scores,axis=-1,keepdims=True)

    attn_weights=np.exp(attn_scores-mx)/np.sum(np.exp(attn_scores-mx),axis=-1,keepdims=True)

    out=attn_weights @ V

    return (out,attn_weights)
    



