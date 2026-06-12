# Inference Benchmarking Suite

This module benchmarks autoregressive decoding performance and KV-cache efficiency for modern LLM attention mechanisms.

Benchmarks include:
- Decode latency
- Prefill latency
- Tokens/sec
- KV-cache memory scaling
- Batch inference throughput
- GQA vs MQA vs standard attention inference behavior

---

# Files

| File | Purpose |
|---|---|
| benchmark_inference.py | Main inference benchmark |
| decode.py | Inference attention implementation |
| kv_cache.py | KV-cache implementation |
| sampling.py | Sampling strategies |

---

# Run Benchmark

```bash
python benchmarks/inference/benchmark_inference.py
```

---

# Outputs

Generated automatically:

```txt
results/
│
├── inference_benchmark.csv
├── decode_latency.png
├── kv_cache_memory.png
└── throughput.png
```