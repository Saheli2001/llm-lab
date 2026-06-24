import torch
import math
import torch.nn as nn
import torch.nn.functional as F

def precompute_freqs_cis(dim, end, theta=10000.0, device='cpu'):
    # theta values, we know theta = 10000^(2(i-1)/d) for i = 1,2,...,d/2
    freqs = 1.0/(theta**(torch.arange(0,dim,2,device=device)))
    # these are the m, n values, the position of the embeddings
    t = torch.arange(end, device=device)
    # to calculate the rotation matrix we need m*theta_i, so each m is multiplied with theta_i, 
    # for i = 1,2,...,d/2, for m = 0,1,...,end-1, for that, we use torch.outer
    freqs = torch.outer(t, freqs)
    #these freqs are m*theta_i
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    # cos , sin gives the elements of rotation matrix R
    return cos, sin

def apply_rope(x, cos, sin):
    # x: (B,T,H,D), this is query/ key
    # x1 denotes even indices corresponding to last dimension of x
    x1 = x[..., ::2]
    # x2 denotes odd indices corresponding to last dimension of x
    x2 = x[..., 1::2]

    # cos and sin are precomputed for the maximum sequence length,
    # so we need to slice them to match the current sequence length
    # suppose cos and sin have shape (T, D/2), we need to unsqueeze them to (1, T, 1, D/2) for broadcasting
    cos = cos[: x.shape[1], :].unsqueeze(0).unsqueeze(2)
    sin = sin[: x.shape[1], :].unsqueeze(0).unsqueeze(2)
    # Now we can apply the rotation to the even and odd parts of x, shape of x_rotated will be (B,T,H,D/2,2)
    x_rotated = torch.stack([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos,
    ], dim=-1) # the stack is creating a new dimension at the end.
    return x_rotated.flatten(-2) #flattens the last two dimensions to get back to the shape (B, T, H, D) 

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        #learnable paraeters , works as the lambda factor in Layernorm
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps) #rmsnorm = x/sqrt(mean(x^2) + eps)
        return self.weight * x

class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class ModernAttention(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, cos, sin):
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if self.n_kv_heads != self.n_heads:
            repeat_factor = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(repeat_factor, dim=2)
            v = v.repeat_interleave(repeat_factor, dim=2)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)
    
class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads, hidden_dim, dropout):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)
        self.attn = ModernAttention(dim, n_heads, n_kv_heads, dropout)
        self.ffn = SwiGLU(dim, hidden_dim)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x
        

class ModernMiniGPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        block_size,
        dim=512,
        n_layers=8,
        n_heads=8,
        n_kv_heads=2,
        hidden_dim=1536,
        dropout=0.0,
    ):
        
        super().__init__()
        self.block_size = block_size
        self.dim = dim

        self.token_embedding = nn.Embedding(vocab_size, dim)

        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads, n_kv_heads, hidden_dim, dropout)
            for _ in range(n_layers)
        ])

        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.block_size

        x = self.token_embedding(idx)
        cos, sin = precompute_freqs_cis(self.dim // 8, T, device=idx.device)

        for layer in self.layers:
            x = layer(x, cos, sin)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )

        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)

        return idx


    
