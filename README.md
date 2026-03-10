# autoresearch

![teaser](progress.png)

*One day, frontier AI research used to be done by meat computers in between eating, sleeping, having other fun, and synchronizing once in a while using sound wave interconnect in the ritual of "group meeting". That era is long gone. Research is now entirely the domain of autonomous swarms of AI agents running across compute cluster megastructures in the skies. The agents claim that we are now in the 10,205th generation of the code base, in any case no one could tell if that's right or wrong as the "code" is now a self-modifying binary that has grown beyond human comprehension. This repo is the story of how it all began. -@karpathy, March 2026*.

The idea: give an AI agent a small but real LLM training setup and let it experiment autonomously overnight. It modifies the code, trains for 5 minutes, checks if the result improved, keeps or discards, and repeats. You wake up in the morning to a log of experiments and (hopefully) a better model. The training code is a lightweight CPU-first transformer implementation. The core idea is that you're not touching any of the Python files like you normally would as a researcher. Instead, you are programming the `program.md` Markdown files that provide context to the AI agents and set up your autonomous research org. The default `program.md` in this repo is intentionally kept as a bare bones baseline, though it's obvious how one would iterate on it over time to find the "research org code" that achieves the fastest research progress, how you'd add more agents to the mix, etc. A bit more context on this project is here in this [tweet](https://x.com/karpathy/status/2029701092347630069).

## How it works

The repo is deliberately kept small and only really has three files that matter:

- **`prepare.py`** — fixed constants, one-time data download, a built-in byte-level tokenizer (no training needed), and runtime utilities (dataloader, evaluation). Not modified by the agent.
- **`train.py`** — the single file the agent edits. Contains a GPT model, AdamW optimizer, and training loop. Everything is fair game: architecture, hyperparameters, optimizer, batch size, etc. **This file is edited and iterated on by the agent**.
- **`program.md`** — baseline instructions for one agent. Point your agent here and let it go. **This file is edited and iterated on by the human**.

By design, training runs for a **fixed 5-minute time budget** (wall clock, excluding startup/compilation), regardless of the details of your compute. The metric is **val_bpb** (validation bits per byte) — lower is better, and vocab-size-independent so architectural changes are fairly compared.

## Quick start

**Requirements:** Python 3.10+, [uv](https://docs.astral.sh/uv/). No GPU required — runs entirely on CPU (or MPS/CUDA if available).

```bash

# 1. Install uv project manager (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Download TinyStories data (one-time, ~1 GB)
uv run prepare.py

# 4. Manually run a single training experiment (~5 min)
uv run train.py
```

If the above commands all work ok, your setup is working and you can go into autonomous research mode.

## Running the agent

Simply spin up your Claude/Codex or whatever you want in this repo (and disable all permissions), then you can prompt something like:

```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

The `program.md` file is essentially a super lightweight "skill".

## Project structure

```
prepare.py      — constants, data download + runtime utilities (do not modify)
train.py        — model, optimizer, training loop (agent modifies this)
program.md      — agent instructions
pyproject.toml  — dependencies
```

## What changed from the original

This fork adapts the original GPU-only autoresearch setup to run on **CPU (and MPS/CUDA if available)**. Key changes:

- **Dataset**: Switched from `climbmix-400b-shuffle` (massive GPU-scale corpus) to [TinyStories](https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean) — short GPT-4 generated stories, ~1 GB total, ideal for small models on modest hardware.
- **Tokenizer**: Replaced the trained BPE tokenizer (`rustbpe` + `tiktoken`) with a **built-in byte-level tokenizer** (vocab_size=256). No tokenizer training step needed — `prepare.py` just downloads data.
- **Model**: Simplified GPT using `F.scaled_dot_product_attention` (PyTorch native, works on CPU/MPS/CUDA). Removed Flash Attention 3, value embeddings, and the sliding window attention pattern. Defaults: `DEPTH=4`, `HEAD_DIM=64`, `MAX_SEQ_LEN=256`.
- **Optimizer**: Replaced `MuonAdamW` (GPU-only, requires `torch.compile`) with plain **AdamW** (`torch.optim.AdamW`).
- **Batch size**: Reduced `TOTAL_BATCH_SIZE` from `2**19` (~524K tokens) to `2**12` (~4K tokens) to suit CPU throughput.
- **Dependencies**: Removed `kernels`, `rustbpe`, `tiktoken`, and the `pytorch-cu128` CUDA index. Now only requires `torch>=2.2.0`, `pyarrow`, `requests`, `numpy`, `pandas`, and `matplotlib`.
- **Output**: `peak_vram_mb` reports `0.0` on CPU (no VRAM); `mfu_percent` is computed but will be near-zero on non-H100 hardware.

## Design choices

- **Single file to modify.** The agent only touches `train.py`. This keeps the scope manageable and diffs reviewable.
- **Fixed time budget.** Training always runs for exactly 5 minutes, regardless of your specific platform. This means you can expect approx 12 experiments/hour and approx 100 experiments while you sleep.
- **Self-contained.** No external dependencies beyond PyTorch and a few small packages. No distributed training, no complex configs. CPU-first, one file, one metric.
- **CPU-first.** The default configuration targets CPU (or MPS/CUDA if available). It uses the [TinyStories dataset](https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean) — simple GPT-4 stories — and a byte-level tokenizer (vocab_size=256) requiring no tokenizer training. This makes it easy to run on a laptop or any machine without a GPU.

## Platform support

This code runs on **CPU by default**, and also supports MPS (Apple Silicon) and CUDA (NVIDIA GPU). The device is auto-detected at runtime (`DEVICE` is exported from `prepare.py`).

The training setup uses the [TinyStories dataset](https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean) — short GPT-4 generated stories — and a built-in byte-level tokenizer (no tokenizer training step needed). This makes it easy to run meaningful experiments on a laptop or any machine without specialized hardware.

## Notable forks

- [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos) (MacOS)
- [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) (MacOS)
- [jsegov/autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx) (Windows)

## License

MIT
