#import libraries
import os
import sys
import time
import math
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from plot_results import plot_results
from config import Config

#set os path
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

# ============================================================
# Naive Attention
# ============================================================

class NaiveAttention(nn.Module):

    def __init__(self, dim, n_heads):

        super().__init__()

        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)

        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):

        B, T, C = x.shape

        q = self.q_proj(x).view(
            B, T, self.n_heads, self.head_dim
        ).transpose(1, 2)

        k = self.k_proj(x).view(
            B, T, self.n_heads, self.head_dim
        ).transpose(1, 2)

        v = self.v_proj(x).view(
            B, T, self.n_heads, self.head_dim
        ).transpose(1, 2)

        attn = (
            q @ k.transpose(-2, -1)
        ) / math.sqrt(self.head_dim)

        mask = torch.tril(
            torch.ones(T, T, device=x.device)
        )

        attn = attn.masked_fill(
            mask == 0,
            float("-inf")
        )

        attn = F.softmax(attn, dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)

# ============================================================
# Flash Attention
# ============================================================

class FlashAttention(nn.Module):

    def __init__(self, dim, n_heads):

        super().__init__()

        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)

        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):

        B, T, C = x.shape

        q = self.q_proj(x).view(
            B, T, self.n_heads, self.head_dim
        ).transpose(1, 2)

        k = self.k_proj(x).view(
            B, T, self.n_heads, self.head_dim
        ).transpose(1, 2)

        v = self.v_proj(x).view(
            B, T, self.n_heads, self.head_dim
        ).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
        )

        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.out_proj(out)


# ============================================================
# GQA Attention
# ============================================================

class GQAAttention(nn.Module):

    def __init__(
        self,
        dim,
        n_heads,
        n_kv_heads,
    ):

        super().__init__()

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads

        self.head_dim = dim // n_heads

        self.q_proj = nn.Linear(
            dim,
            n_heads * self.head_dim,
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

    def forward(self, x):

        B, T, C = x.shape

        q = self.q_proj(x).view(
            B, T, self.n_heads, self.head_dim
        )

        k = self.k_proj(x).view(
            B, T, self.n_kv_heads, self.head_dim
        )

        v = self.v_proj(x).view(
            B, T, self.n_kv_heads, self.head_dim
        )

        repeat_factor = (
            self.n_heads // self.n_kv_heads
        )

        k = k.repeat_interleave(
            repeat_factor,
            dim=2,
        )

        v = v.repeat_interleave(
            repeat_factor,
            dim=2,
        )

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

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

# ============================================================
# MQA Attention
# ============================================================

class MQAAttention(nn.Module):

    def __init__(
        self,
        dim,
        n_heads,
    ):

        super().__init__()

        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.q_proj = nn.Linear(
            dim,
            dim,
            bias=False,
        )

        # Single KV head
        self.k_proj = nn.Linear(
            dim,
            self.head_dim,
            bias=False,
        )

        self.v_proj = nn.Linear(
            dim,
            self.head_dim,
            bias=False,
        )

        self.out_proj = nn.Linear(
            dim,
            dim,
            bias=False,
        )

    def forward(self, x):

        B, T, C = x.shape

        q = self.q_proj(x).view(
            B, T, self.n_heads, self.head_dim
        )

        k = self.k_proj(x).view(
            B, T, 1, self.head_dim
        )

        v = self.v_proj(x).view(
            B, T, 1, self.head_dim
        )

        k = k.repeat_interleave(
            self.n_heads,
            dim=2,
        )

        v = v.repeat_interleave(
            self.n_heads,
            dim=2,
        )

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

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


# ============================================================
# Sliding Window Attention
# ============================================================

class SlidingWindowAttention(nn.Module):

    def __init__(
        self,
        dim,
        n_heads,
        window_size=128,
    ):

        super().__init__()

        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.window_size = window_size

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)

        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):

        B, T, C = x.shape

        q = self.q_proj(x).view(
            B, T, self.n_heads, self.head_dim
        ).transpose(1, 2)

        k = self.k_proj(x).view(
            B, T, self.n_heads, self.head_dim
        ).transpose(1, 2)

        v = self.v_proj(x).view(
            B, T, self.n_heads, self.head_dim
        ).transpose(1, 2)

        attn = (
            q @ k.transpose(-2, -1)
        ) / math.sqrt(self.head_dim)

        causal_mask = torch.tril(
            torch.ones(T, T, device=x.device)
        )

        window_mask = torch.triu(
            torch.ones(T, T, device=x.device),
            diagonal=-self.window_size,
        )

        mask = causal_mask * window_mask

        attn = attn.masked_fill(
            mask == 0,
            float("-inf"),
        )

        attn = F.softmax(attn, dim=-1)

        out = attn @ v

        out = out.transpose(
            1,
            2,
        ).contiguous().view(B, T, C)

        return self.out_proj(out)


# ============================================================
# Benchmark Function
# ============================================================

def benchmark_attention(
    model,
    seq_len,
    batch_size,
    dim,
    device,
    dtype=torch.float16,
    warmup_iters=10,
    benchmark_iters=50,
):

    model = model.to(device=device, dtype=dtype)

    x = torch.randn(
        batch_size,
        seq_len,
        dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )

    # Warmup
    for _ in range(warmup_iters):

        out = model(x)

        loss = out.mean()

        loss.backward()

    torch.cuda.synchronize()

    # Reset memory stats
    torch.cuda.reset_peak_memory_stats(device)

    # Forward Benchmark
    start_forward = time.time()

    for _ in range(benchmark_iters):

        out = model(x)

    torch.cuda.synchronize()

    forward_time = (
        time.time() - start_forward
    ) / benchmark_iters

    # Backward Benchmark
    start_backward = time.time()

    for _ in range(benchmark_iters):

        out = model(x)

        loss = out.mean()

        loss.backward()

    torch.cuda.synchronize()

    backward_time = (
        time.time() - start_backward
    ) / benchmark_iters

 
    # Memory
   peak_memory = (
        torch.cuda.max_memory_allocated(device)
        / (1024 ** 3)
    )

    # Throughput
    tokens_per_second = (
        batch_size * seq_len
    ) / forward_time

    return {
        "forward_time_ms": forward_time * 1000,
        "backward_time_ms": backward_time * 1000,
        "tokens_per_second": tokens_per_second,
        "peak_memory_gb": peak_memory,
    }

# Main

def main():

    assert torch.cuda.is_available(), \
        "CUDA GPU required."

    device = "cuda:3"

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    dim = 512
    n_heads = 8

    batch_sizes = [8, 16]
    seq_lengths = [128, 256, 512, 1024*4]

    results = []

    attention_variants = {

        "naive": NaiveAttention(
            dim,
            n_heads,
        ),
    
        "flash": FlashAttention(
            dim,
            n_heads,
        ),
    
        "gqa": GQAAttention(
            dim,
            n_heads,
            n_kv_heads=2,
        ),
    
        "mqa": MQAAttention(
            dim,
            n_heads,
        ),
    
        "sliding_window": SlidingWindowAttention(
            dim,
            n_heads,
            window_size=128,
        ),
    }

    for name, model in attention_variants.items():

        print("\n" + "=" * 60)
        print(f"Benchmarking: {name}")
        print("=" * 60)

        for batch_size in batch_sizes:

            for seq_len in seq_lengths:

                stats = benchmark_attention(
                    model=model,
                    seq_len=seq_len,
                    batch_size=batch_size,
                    dim=dim,
                    device=device,
                )

                result = {
                    "attention_type": name,
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    **stats,
                }

                results.append(result)

                print(
                    f"[{name}] "
                    f"B={batch_size} "
                    f"T={seq_len} | "
                    f"Forward: {stats['forward_time_ms']:.2f} ms | "
                    f"Backward: {stats['backward_time_ms']:.2f} ms | "
                    f"TPS: {stats['tokens_per_second']:.2f} | "
                    f"VRAM: {stats['peak_memory_gb']:.2f} GB"
                )

    # Save CSV
    df = pd.DataFrame(results)

    os.makedirs("benchmark_results", exist_ok=True)

    csv_path = (
        "benchmark_results/attention_benchmark.csv"
    )

    df.to_csv(csv_path, index=False)

    print("\nSaved benchmark results:")
    print(csv_path)

    print("\nFinal Results:")
    print(df)

    plot_results(df)

if __name__ == "__main__":

    main()