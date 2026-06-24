# LLM-Lab

A research-oriented repository for implementing, benchmarking, and understanding modern Large Language Models from first principles.

The goal of this project is not to use existing frameworks as black boxes, but to build and analyze the core components behind modern LLMs including Transformers, GPT-style architectures, efficient attention mechanisms, parameter-efficient fine-tuning methods, alignment algorithms, quantization techniques, and state-space models.

> This repository is actively under development.

---

## Current Features

### Modern GPT Implementation

Implemented from scratch in PyTorch:

* Rotary Positional Embeddings (RoPE)
* RMSNorm
* SwiGLU Feed Forward Networks
* Flash Attention (PyTorch SDPA)
* Grouped Query Attention (GQA)
* Multi Query Attention (MQA)
* Weight Tying
* Autoregressive Text Generation

---

## Benchmarking Suite

### Attention Benchmarking

Implemented attention benchmarking for:

* Standard Multi-Head Attention (MHA)
* Flash Attention
* Grouped Query Attention (GQA)
* Multi Query Attention (MQA)
* Sliding Window Attention

Metrics:

* Throughput
* Memory Usage
* Sequence Length Scaling

Outputs:

* CSV Reports
* Throughput Plots
* Memory Scaling Plots

---

### Inference Benchmarking

Implemented inference benchmarking for:

* KV Cache Scaling
* Decode Latency
* Prefill Latency
* Tokens per Second
* Memory Usage

Compared:

* MHA
* GQA
* MQA

Outputs:

* Inference Reports
* Throughput Analysis
* KV Cache Scaling Visualizations

---

## Repository Structure

```text
llm-lab/

├── models/
│   ├── gpt/
│   ├── llama/
│   └── mamba/
│
├── training/
├── inference/
├── finetuning/
├── alignment/
├── quantization/
│
├── benchmarks/
│   ├── attention/
│   ├── inference/
│   ├── training/
│   └── profiler/
│
├── visualization/
├── reports/
├── experiments/
└── tests/
```

---

## Roadmap

### Core Models

* [x] GPT-style Transformer
* [ ] LLaMA Architecture
* [ ] Mamba Architecture

### Efficient Attention

* [x] Flash Attention
* [x] GQA
* [x] MQA
* [x] Sliding Window Attention
* [ ] Paged Attention

### Fine-Tuning

* [ ] LoRA
* [ ] QLoRA
* [ ] Adapter Tuning

### Alignment

* [ ] DPO
* [ ] GRPO
* [ ] PPO
* [ ] Reward Modeling

### Quantization

* [ ] INT8 Quantization
* [ ] NF4 Quantization
* [ ] GPTQ

### Benchmarking

* [x] Attention Benchmarking
* [x] Inference Benchmarking
* [ ] Training Benchmarking
* [ ] Profiler Suite

### Visualization

* [ ] Attention Maps
* [ ] RoPE Visualizations
* [ ] Token Probability Analysis

---

## Motivation

Most LLM repositories focus on training models with existing libraries.

This project focuses on understanding:

* Why modern architectures work
* How attention mechanisms scale
* How inference systems are optimized
* How parameter-efficient fine-tuning works
* How alignment methods are implemented
* How quantization reduces deployment cost

The emphasis is on first-principles implementation, experimentation, and reproducible benchmarking.

---

## Status

Current Stage:

* Core GPT implementation completed
* Attention benchmarking completed
* Inference benchmarking completed
* LoRA / QLoRA implementation in progress

More modules and benchmark reports will be added as development continues.
