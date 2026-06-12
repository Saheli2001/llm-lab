import os
import time

import torch
import pandas as pd
import matplotlib.pyplot as plt

from decode import InferenceAttention
from kv_cache import KVCache


DEVICE = "cuda:2"
DTYPE = torch.float16


def benchmark_decode(
    model,
    seq_len,
    decode_steps,
    batch_size,
    dim,
    n_kv_heads,
):

    model = model.to(DEVICE).to(DTYPE)

    kv_cache = KVCache(
        batch_size=batch_size,
        max_seq_len=seq_len + decode_steps,
        n_kv_heads=n_kv_heads,
        head_dim=dim // 8,
        device=DEVICE,
    )

    # ========================================================
    # Prefill
    # ========================================================

    x = torch.randn(
        batch_size,
        seq_len,
        dim,
        device=DEVICE,
        dtype=DTYPE,
    )

    torch.cuda.synchronize()

    start_prefill = time.time()

    with torch.no_grad():

        for pos in range(seq_len):

            token = x[:, pos:pos+1]

            _ = model(
                token,
                kv_cache=kv_cache,
                position=pos,
            )

    torch.cuda.synchronize()

    prefill_time = (
        time.time() - start_prefill
    )

    # ========================================================
    # Decode Benchmark
    # ========================================================

    decode_input = torch.randn(
        batch_size,
        1,
        dim,
        device=DEVICE,
        dtype=DTYPE,
    )

    torch.cuda.reset_peak_memory_stats()

    torch.cuda.synchronize()

    start_decode = time.time()

    with torch.no_grad():

        for i in range(decode_steps):

            _ = model(
                decode_input,
                kv_cache=kv_cache,
                position=seq_len + i,
            )

    torch.cuda.synchronize()

    decode_time = (
        time.time() - start_decode
    )

    peak_memory = (
        torch.cuda.max_memory_allocated()
        / (1024 ** 3)
    )

    tokens_per_second = (
        decode_steps * batch_size
    ) / decode_time

    latency_per_token_ms = (
        decode_time / decode_steps
    ) * 1000

    return {
        "prefill_time_s": prefill_time,
        "decode_time_s": decode_time,
        "latency_per_token_ms": latency_per_token_ms,
        "tokens_per_second": tokens_per_second,
        "peak_memory_gb": peak_memory,
    }


def plot_results(df):

    os.makedirs("results", exist_ok=True)

    # ========================================================
    # Decode Latency
    # ========================================================

    plt.figure(figsize=(8, 6))

    for attn in df["attention_type"].unique():

        subset = df[
            df["attention_type"] == attn
        ]

        plt.plot(
            subset["seq_len"],
            subset["latency_per_token_ms"],
            marker="o",
            label=attn,
        )

    plt.xlabel("Sequence Length")
    plt.ylabel("Latency / Token (ms)")
    plt.title("Decode Latency Scaling")

    plt.legend()

    plt.savefig(
        "results/decode_latency.png"
    )

    plt.close()

    # ========================================================
    # KV Cache Memory
    # ========================================================

    plt.figure(figsize=(8, 6))

    for attn in df["attention_type"].unique():

        subset = df[
            df["attention_type"] == attn
        ]

        plt.plot(
            subset["seq_len"],
            subset["peak_memory_gb"],
            marker="o",
            label=attn,
        )

    plt.xlabel("Sequence Length")
    plt.ylabel("Peak Memory (GB)")
    plt.title("KV Cache Memory Scaling")

    plt.legend()

    plt.savefig(
        "results/kv_cache_memory.png"
    )

    plt.close()

    # ========================================================
    # Throughput
    # ========================================================

    plt.figure(figsize=(8, 6))

    for attn in df["attention_type"].unique():

        subset = df[
            df["attention_type"] == attn
        ]

        plt.plot(
            subset["seq_len"],
            subset["tokens_per_second"],
            marker="o",
            label=attn,
        )

    plt.xlabel("Sequence Length")
    plt.ylabel("Tokens / Second")
    plt.title("Inference Throughput")

    plt.legend()

    plt.savefig(
        "results/throughput.png"
    )

    plt.close()


def main():

    assert torch.cuda.is_available()

    dim = 512
    n_heads = 8

    batch_size = 1

    seq_lengths = [
        128,
        256,
        512,
        1024,
        2048,
        4096,
    ]

    decode_steps = 128

    attention_variants = {

        "mha": InferenceAttention(
            dim,
            n_heads,
            n_kv_heads=8,
        ),

        "gqa": InferenceAttention(
            dim,
            n_heads,
            n_kv_heads=2,
        ),

        "mqa": InferenceAttention(
            dim,
            n_heads,
            n_kv_heads=1,
        ),
    }

    results = []

    for name, model in attention_variants.items():

        print("\n" + "=" * 60)
        print(f"Benchmarking: {name}")
        print("=" * 60)

        for seq_len in seq_lengths:

            stats = benchmark_decode(
                model=model,
                seq_len=seq_len,
                decode_steps=decode_steps,
                batch_size=batch_size,
                dim=dim,
                n_kv_heads=model.n_kv_heads,
            )

            result = {
                "attention_type": name,
                "seq_len": seq_len,
                **stats,
            }

            results.append(result)

            print(
                f"[{name}] "
                f"T={seq_len} | "
                f"Latency/token: "
                f"{stats['latency_per_token_ms']:.3f} ms | "
                f"TPS: "
                f"{stats['tokens_per_second']:.2f} | "
                f"VRAM: "
                f"{stats['peak_memory_gb']:.2f} GB"
            )

    df = pd.DataFrame(results)

    os.makedirs("results", exist_ok=True)

    df.to_csv(
        "results/inference_benchmark.csv",
        index=False,
    )

    plot_results(df)

    print("\nSaved benchmark results.")


if __name__ == "__main__":

    main()