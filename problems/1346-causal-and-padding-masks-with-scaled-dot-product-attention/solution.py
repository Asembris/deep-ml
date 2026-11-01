import torch
import torch.nn as nn
import torch.nn.functional as F


def attention(q, k, v, key_padding_mask=None, causal=False):
    # TODO: build a single bool attn_mask broadcastable to (B, H, T_q, T_k)
    #       where True means "may attend"
    # TODO: combine the causal lower-triangular mask and the key padding mask
    # TODO: return F.scaled_dot_product_attention(q, k, v, attn_mask=...)
    B, H, T_q, d = q.shape
    T_k = k.shape[-2]

    mask = torch.ones(
        (B, 1, T_q, T_k),
        dtype=torch.bool,
        device=q.device
    )

    if causal:
        causal_mask = torch.tril(
            torch.ones((T_q, T_k), dtype=torch.bool, device=q.device)
        )
        mask = mask & causal_mask

    if key_padding_mask is not None:
        padding_mask =  key_padding_mask.bool()
        mask = mask & padding_mask[:, None, None, :]

    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=mask
    )
