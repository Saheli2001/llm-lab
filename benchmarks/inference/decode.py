import torch
import torch.nn as nn
import torch.nn.functional as F


class InferenceAttention(nn.Module):

    def __init__(
        self,
        dim,
        n_heads,
        n_kv_heads=None,
    ):

        super().__init__()

        self.n_heads = n_heads

        if n_kv_heads is None:
            n_kv_heads = n_heads

        self.n_kv_heads = n_kv_heads

        self.head_dim = dim // n_heads

        self.q_proj = nn.Linear(
            dim,
            dim,
            bias=False,
        )

        self.k_proj = nn.Linear(
            dim,
            n_kv_heads * self.head_dim,
            bias=False,
        )

        self.v_proj = nn.Linear(
            dim,
            n_kv_heads * self.head_dim,
            bias=False,
        )

        self.out_proj = nn.Linear(
            dim,
            dim,
            bias=False,
        )

    def forward(
        self,
        x,
        kv_cache=None,
        position=None,
    ):

        B, T, C = x.shape

        q = self.q_proj(x).view(
            B,
            T,
            self.n_heads,
            self.head_dim,
        )

        k = self.k_proj(x).view(
            B,
            T,
            self.n_kv_heads,
            self.head_dim,
        )

        v = self.v_proj(x).view(
            B,
            T,
            self.n_kv_heads,
            self.head_dim,
        )

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # ====================================================
        # KV Cache Update
        # ====================================================

        if kv_cache is not None:

            kv_cache.update(
                k,
                v,
                position,
            )

            k, v = kv_cache.get_cache(
                position + 1
            )

        # ====================================================
        # GQA / MQA Expansion
        # ====================================================

        if self.n_kv_heads != self.n_heads:

            repeat_factor = (
                self.n_heads // self.n_kv_heads
            )

            k = k.repeat_interleave(
                repeat_factor,
                dim=1,
            )

            v = v.repeat_interleave(
                repeat_factor,
                dim=1,
            )

        # ====================================================
        # Attention
        # ====================================================

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
        )

        out = out.transpose(
            1,
            2,
        ).contiguous().view(B, T, C)

        return self.out_proj(out)