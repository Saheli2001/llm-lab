# Benchmarking Suite

This folder contains benchmarking and profiling utilities for analyzing the performance characteristics of modern LLM components and training/inference workloads.

The goal of this benchmarking suite is to study practical systems-level behavior of transformer architectures under different efficiency optimizations and scaling settings.

---

# Current Benchmarks

## Attention Benchmark
`benchmark_attention.py`

Benchmarks multiple attention implementations across varying:
- sequence lengths
- batch sizes
- attention variants

Measured metrics:
- forward latency
- backward latency
- throughput (tokens/sec)
- peak VRAM usage

Implemented attention mechanisms:
- Naive Attention
- FlashAttention
- Grouped Query Attention (GQA)
- Multi-Query Attention (MQA)
- Sliding Window Attention

---

# Planned Benchmarks

## Training Benchmark
`benchmark_training.py`

Planned features:
- full training step profiling
- mixed precision comparison
- optimizer overhead analysis
- throughput scaling
- memory profiling

---

## Inference Benchmark
`benchmark_inference.py`

Planned features:
- KV-cache benchmarking
- decode latency analysis
- autoregressive throughput
- cache memory comparison
- streaming generation profiling

---

## Profiler
`profiler.py`

Planned features:
- PyTorch profiler integration
- CUDA kernel tracing
- Chrome trace export
- bottleneck analysis

---

# Benchmark Outputs

Generated benchmark artifacts include:
- CSV logs
- throughput graphs
- memory usage graphs
- profiling traces

Outputs are stored under:

```txt
benchmark_results/
```

---

# Running Benchmarks

## Attention Benchmark

```bash
python benchmarks/benchmark_attention.py
```

---

# Dependencies

```bash
pip install torch pandas matplotlib
```

---

# Design Philosophy

The benchmarks are intentionally modular and component-isolated.

Rather than benchmarking only full-model training, individual subsystems are profiled independently to better understand:
- architectural bottlenecks
- scaling behavior
- memory efficiency
- inference optimizations

This structure makes it easier to:
- compare attention implementations
- study optimization tradeoffs
- reproduce experiments
- extend benchmarking coverage

---

# Reports & Findings

Detailed experimental findings, graphs, and scaling observations are documented separately under:

```txt
reports/
```

Example:
```txt
reports/attention_scaling.md
```