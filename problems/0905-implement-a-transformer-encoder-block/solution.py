import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # TODO: norm1, attn, norm2, mlp, dropout
        self.norm1=nn.LayerNorm(d_model)
        self.attn=nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm2=nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model,d_ff),
            nn.GELU(),
            nn.Linear(d_ff,d_model)
        )

        self.dropout=nn.Dropout(dropout)

    def forward(self, x):
        # TODO: pre-LN attention sublayer, then pre-LN MLP sublayer, both with residual
        h=self.norm1(x)
        x1=x+self.dropout(self.attn(h,h,h)[0])
        x2=x1+self.dropout(self.ffn(self.norm2(x1)))
        return x2




