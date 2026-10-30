import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        # TODO: store d_head, create q/k/v/out projections (all bias=False)
        assert d_model%num_heads==0, "not divisible"
        self.d_head=d_model//num_heads
        self.q_proj=nn.Linear(d_model,d_model,bias=False)
        self.k_proj=nn.Linear(d_model,d_model,bias=False)
        self.v_proj=nn.Linear(d_model,d_model,bias=False)
        self.out_proj=nn.Linear(d_model,d_model,bias=False)

    def forward(self, x, mask=None):
        # x: (B, T, d_model); mask: (T, T) of 0 and -inf, or None
        # TODO: project, reshape into heads, scaled dot-product, mask, softmax, combine, reshape back, out_proj
        B,T,d_model=x.shape
        Q=self.q_proj(x)
        K=self.k_proj(x)
        V=self.v_proj(x)
        Q=Q.reshape(B,T,d_model//self.d_head,self.d_head)
        K=K.reshape(B,T,d_model//self.d_head,self.d_head)
        V=V.reshape(B,T,d_model//self.d_head,self.d_head)
        Q=Q.transpose(1,2)
        K=K.transpose(1,2)
        V=V.transpose(1,2)

        attn_scores=(Q @ K.transpose(-2, -1))/(self.d_head**0.5)
        if mask is not None:
            attn_scores+=mask

        attn_weights=F.softmax(attn_scores, dim=-1)
        out=attn_weights @ V
        out=out.transpose(1,2)
        out=out.reshape(-1)
        out=out.reshape(B,T,d_model)
        return out




