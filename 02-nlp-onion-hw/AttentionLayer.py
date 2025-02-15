import torch.nn as nn


class AttentionLayer(nn.Module):
    def __init__(self, embed_dim, n_heads=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.multihead_attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)

    def forward(self, x, attn_mask=None):
        attn_output, _ = self.multihead_attn(x, x, x, attn_mask=attn_mask)
        return attn_output
