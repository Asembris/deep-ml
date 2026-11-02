import numpy as np

def attention_dropout(attn_weights, values, dropout_rate, mask):
    """Apply dropout mask to attention weights and compute context vectors."""
    
    attn_weights=np.where(mask,attn_weights,0)
    attn_weights/=(1-dropout_rate)

    context=attn_weights @ values
    return context.tolist()

