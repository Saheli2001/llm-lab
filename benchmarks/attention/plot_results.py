# ============================================================
# Plot Results
# ============================================================

import matplotlib.pyplot as plt
import os


def plot_results(df):

    os.makedirs("benchmark_results", exist_ok=True)

    # --------------------------------------------------------
    # Tokens/sec vs Sequence Length
    # --------------------------------------------------------

    plt.figure(figsize=(8, 6))

    for attn_type in df["attention_type"].unique():

        subset = df[
            (df["attention_type"] == attn_type)
            &
            (df["batch_size"] == 8)
        ]

        plt.plot(
            subset["seq_len"],
            subset["tokens_per_second"],
            marker="o",
            label=attn_type,
        )

    plt.xlabel("Sequence Length")
    plt.ylabel("Tokens / Second")
    plt.title("Naive vs Flash Attention Throughput")

    plt.legend()

    throughput_path = (
        "benchmark_results/throughput_comparison.png"
    )

    plt.savefig(throughput_path)

    print(f"\nSaved plot: {throughput_path}")

    plt.close()

    # --------------------------------------------------------
    # Peak VRAM vs Sequence Length
    # --------------------------------------------------------

    plt.figure(figsize=(8, 6))

    for attn_type in df["attention_type"].unique():

        subset = df[
            (df["attention_type"] == attn_type)
            &
            (df["batch_size"] == 8)
        ]

        plt.plot(
            subset["seq_len"],
            subset["peak_memory_gb"],
            marker="o",
            label=attn_type,
        )

    plt.xlabel("Sequence Length")
    plt.ylabel("Peak VRAM (GB)")
    plt.title("Naive vs Flash Attention Memory Usage")

    plt.legend()

    memory_path = (
        "benchmark_results/memory_comparison.png"
    )

    plt.savefig(memory_path)

    print(f"Saved plot: {memory_path}")

    plt.close()