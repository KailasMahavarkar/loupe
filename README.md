<div align="center">

# 🔍 loupe

**Running [karpathy/autoresearch](https://github.com/karpathy/autoresearch) on a consumer RTX 3060 - setup, fixes, and optimizations that got us from 1.774 → 1.606 val_bpb**

[![GPU](https://img.shields.io/badge/GPU-RTX%203060%2012GB-76B900?logo=nvidia&logoColor=white)](https://www.nvidia.com)
[![CUDA](https://img.shields.io/badge/CUDA-12.8-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9.1-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://python.org)
[![uv](https://img.shields.io/badge/uv-package%20manager-DE5FE9)](https://docs.astral.sh/uv/)
[![OS](https://img.shields.io/badge/Ubuntu-24.04-E95420?logo=ubuntu&logoColor=white)](https://ubuntu.com)

</div>

---

## 🧠 What is autoresearch?

[autoresearch](https://github.com/karpathy/autoresearch) by Andrej Karpathy lets an AI agent autonomously conduct LLM research while you sleep. The loop:

1. An AI agent (Claude, Cursor, etc.) reads `train.py`
2. Tweaks something - a learning rate, attention pattern, layer depth
3. Runs a 5-minute training experiment on your GPU
4. Checks the score (`val_bpb` - lower = smarter model)
5. Keeps improvements, reverts failures - repeat 50-100x overnight

The models are small (toy scale), but the *process* is the point - AI doing the grunt work of ML research.

---

## 🖥️ System

| Component | Spec |
|-----------|------|
| GPU | NVIDIA GeForce RTX 3060 (Ampere, SM 8.6) |
| VRAM | 12 GB |
| RAM | 29 GB |
| OS | Ubuntu 24.04 (native, not WSL2) |
| Driver | 590.48.01 |
| Python | 3.10 (managed by uv) |

---

## 🔧 Setup Steps

### 1 - Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/share/../bin/env
```

### 2 - Clone and install

```bash
git clone https://github.com/karpathy/autoresearch.git
cd autoresearch
uv sync   # downloads PyTorch + all deps (~2.5 GB, ~3 min)
```

### 3 - Prepare data

```bash
uv run prepare.py   # downloads shards, trains BPE tokenizer (~2 min)
```

### 4 - Apply RTX 3060 fixes to train.py

The upstream code uses FlashAttention-3, which requires Hopper GPUs (H100). Ampere (RTX 3060, SM 8.6) needs PyTorch's built-in SDPA instead.

**Fix 1 - Replace FA3 with SDPA** in `CausalSelfAttention.forward()`:

```python
# Remove this line at the top:
fa3 = get_kernel(repo).flash_attn_interface

# Replace with:
fa3 = None
```

```python
# Replace the attention call:
# y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)

# With PyTorch SDPA:
q = q.transpose(1, 2)
k = k.transpose(1, 2)
v = v.transpose(1, 2)
y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
y = y.transpose(1, 2)
```

**Fix 2 - Set batch size** (upstream default 128 OOMs on 12 GB):

```python
DEVICE_BATCH_SIZE = 16  # safe with torch.compile on RTX 3060
```

### 5 - Run

```bash
uv run train.py
```

---

## ⚡ Optimizations

These are applied **on top** of the fixes above, in order of impact.

### 🥇 torch.compile - biggest win

The upstream + WSL2 fork disabled `torch.compile` because Inductor/Triton fails on Ampere under WSL2. On **native Ubuntu it works perfectly** and gives a massive speedup.

Re-enable all three compile points:

```python
# Optimizer step functions - fuses kernel launches
@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(...):

@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(...):

# Full model
model = torch.compile(model, dynamic=False, fullgraph=True)
```

`fullgraph=True` prevents silent graph breaks - forces the entire model into one compiled graph.

### 🥈 GPU power limit + locked clocks

By default the RTX 3060 is capped at 170W and clocks vary under load. Locking them eliminates jitter and squeezes more out of sustained training:

```bash
sudo nvidia-smi -pm 1           # persistence mode
sudo nvidia-smi -pl 200         # raise power limit to max (200W)
sudo nvidia-smi -lgc 2115       # lock SM clocks to max (2115 MHz)
```

Result: step time goes from variable (6400-6500ms) to rock-solid 6345ms every step.

### 🥉 TF32 + Flash SDPA flags

```python
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
```

These ensure PyTorch picks the fastest available kernels and uses TF32 for matmuls (Ampere supports this natively).

---

## 📊 Results Progression

Every run is 5 minutes (300s time budget). Lower val_bpb = better model quality.

| Config | val_bpb | tok/sec | Steps | Peak VRAM |
|--------|---------|---------|-------|-----------|
| Baseline (upstream defaults, batch 8) | 1.774 | 47k | 38 | 6.2 GB |
| + batch 16 (no compile) | OOM | - | - | >12 GB |
| + torch.compile, batch 16 | 1.613 | 81k | 58 | 6.1 GB |
| **+ fullgraph, 200W, locked clocks** | **1.606** | **83k** | **59** | **6.1 GB** |

**72% throughput improvement** (47k → 83k tok/sec) and **9.5% better score** (1.774 → 1.606) compared to baseline.

---

## ❌ Things Tried That Didn't Help

<details>
<summary>Gradient checkpointing</summary>

Tried to fit batch 16 before we had torch.compile by checkpointing all 8 layers. VRAM dropped to 5.4 GB but step time went from 11s → 14s - fewer steps in the 5 min budget = worse score.

| | val_bpb | Steps |
|-|---------|-------|
| Batch 8, no ckpt | 1.774 | 38 |
| Batch 16, full ckpt | 1.815 | 33 |
| Batch 16, half ckpt (4/8 layers) | 1.798 | 35 |

</details>

<details>
<summary>CUDA graphs (reduce-overhead compile mode)</summary>

`torch.compile(model, mode="reduce-overhead")` uses CUDA graphs which require stable memory addresses. The embedding lookup `wte(idx)` with changing inputs causes a `CUDAGraphs overwrite` error. Not compatible with this model's data flow without significant refactoring.

</details>

<details>
<summary>Bigger model (DEPTH=12)</summary>

With 6 GB VRAM headroom after compile, tried DEPTH=12 (n_embd=768, 135M params). OOMed at 11.2 GB. DEPTH=10 fit (9.4 GB, 86M params) but got only 25 steps in 300s vs 59 steps at DEPTH=8 - the time budget heavily penalises slower models.

</details>

<details>
<summary>Larger total batch size (2^20)</summary>

Doubling `TOTAL_BATCH_SIZE` from 2^19 to 2^20 doubles gradient accumulation steps from 16 to 32 - better gradient quality per step but half as many optimizer steps in 300s. Net result: worse score for fixed time budget.

</details>

<details>
<summary>Numpy dataloader optimizations</summary>

Replaced the Python list `doc_buffer` with numpy int64 arrays and vectorised the best-fit packing search with `np.argmax`. Correctness validated - shapes, BOS alignment, input/target shift all correct. However `np.argmax` ties differently than the Python loop, changing document ordering and producing slightly different (worse) val_bpb. Reverted since the dataloader is not the bottleneck (GPU-bound at 6.3s/step).

</details>

---

## 🏁 Final Config (train.py)

```python
# Architecture
DEPTH = 8
ASPECT_RATIO = 64
HEAD_DIM = 128
WINDOW_PATTERN = "SSSL"

# Batch / tokens
TOTAL_BATCH_SIZE = 2**19   # ~524K tokens/step
DEVICE_BATCH_SIZE = 16     # fits 12 GB with compile

# Compile
model = torch.compile(model, dynamic=False, fullgraph=True)
# @torch.compile(dynamic=False, fullgraph=True) on both optimizer step fns

# Runtime flags
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
```

```bash
# Run (no TORCH_DYNAMO_DISABLE needed on native Ubuntu)
uv run train.py
```

---

## 📝 Notes

- The WSL2 fork ([OpenCnid/autoresearch](https://github.com/OpenCnid/autoresearch)) disables `torch.compile` because Triton/Inductor fails under WSL2 on Ampere. On native Linux this is not an issue.
- CPU governor was already on `performance` - no change needed.
- GNOME Shell uses ~120 MB of VRAM for display. Not worth killing the desktop for the gain.
- val_bpb has ~0.01-0.02 run-to-run variance even with fixed seeds, due to data packing nondeterminism.
