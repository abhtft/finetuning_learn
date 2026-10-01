# Hardware Exploration & Limitations Guide

> **Hardware Profile Analyzed**:  
> * **GPU**: NVIDIA GeForce RTX 5050 Laptop GPU (Blackwell Architecture)
> * **Total VRAM**: 8,151 MiB (~7.933 GB effective, ~7.5 GB usable after OS/WSL overhead)
> * **Compute Precision**: Native Bfloat16 / FP16 / INT8 / INT4 (Tensor Cores)
> * **Power / Thermal Envelope**: 70W Max TDP | ~42W sustained during training | ~76°C operating temp
> * **CUDA Stack**: CUDA 12.8 Toolkit | PyTorch 2.11.0+cu128 | Triton 3.6.0

---

## 1. Real-World Benchmark Analysis (Gemma-4 E4B / 8B Class)

During our live training benchmark of `unsloth/gemma-4-E4B-it` (8.01 Billion parameter multimodal architecture):

| Metric | Measured Value | Practical Interpretation |
| :--- | :--- | :--- |
| **Quantization** | 4-bit QLoRA (BitsAndBytes NF4) | Base model weights compress to ~4.5 GB |
| **LoRA Parameters** | 18.35 Million ($r=8, \alpha=8$) | Trainable parameters account for 0.23% |
| **Max Sequence Length** | 1,024 tokens | Fits within activation budget |
| **Peak VRAM Consumed** | **7,739 MiB / 8,151 MiB (~95%)** | Operating at the ceiling of available memory |
| **Step Speed** | **~13 – 15 seconds / step** | Effective throughput: 0.26 samples/sec |
| **30-Step Run Time** | **7 minutes 41 seconds** | Stable convergence with 0 CUDA OOM crashes |

---

## 2. Capabilities Matrix (What Your RTX 5050 Can Run)

```
       [ 1B - 3B Models ] ────────► 4-bit / 8-bit / 16-bit LoRA | Context up to 4096-8192 | Fast (2-5s/step)
       [ 7B - 8B Models ] ────────► 4-bit QLoRA ONLY | Context up to 1024-2048 | Moderate (12-16s/step)
       [  14B+ Models   ] ────────► Inference Only (4-bit) | Fine-Tuning Not Possible on Single 8GB GPU
```

### Detailed Breakdown

| Model Family | Model Size | Recommended Mode | Max Context (`max_seq_length`) | Batch / Grad Accum | Feasibility |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Llama-3.2 / Qwen-2.5** | 1B – 3B | 4-bit or 16-bit LoRA | **4,096 – 8,192** | `batch_size=2`, `grad_accum=4` |  **Optimal** (Very fast, plenty of headroom) |
| **Gemma-2 / Gemma-3** | 2B | 4-bit or 16-bit LoRA | **4,096** | `batch_size=2`, `grad_accum=4` |  **Optimal** |
| **Gemma-4 / Llama-3.1 / Qwen-2.5** | 7B – 8B | **4-bit QLoRA ONLY** | **1,024** (or 2048 with offload) | `batch_size=1`, `grad_accum=4–16` |  **Supported** (At 90-95% VRAM capacity) |
| **Mistral / Zephyr** | 7B | **4-bit QLoRA ONLY** | **1,024 – 2,048** | `batch_size=1`, `grad_accum=4–8` |  **Supported** |
| **Qwen-2.5 / DeepSeek** | 14B | 4-bit Inference Only | 2,048 (Inference) | N/A (Training OOMs) |  **Inference Only** |
| **Any LLM** | Any Size | **Full Fine-Tuning** | N/A | N/A | ❌ **Not Possible** (Requires 40–80GB+ VRAM) |

---

## 3. Hardware Limitations & Bottlenecks

### 1. Hard VRAM Ceiling (7.93 GB Total)
* **The 8B Limit**: 8B models are the absolute upper boundary for fine-tuning on an 8GB GPU. 
* **14B+ Fine-Tuning Impossible**: In 4-bit, a 14B model requires ~8.5–9.5 GB just for base weights and optimizer buffers, exceeding physical VRAM.
* **No Multimodal Vision Training on 8B**: When fine-tuning Gemma-4 or Llama-3.2-Vision, keep `finetune_vision_layers=False`. Enabling vision adapter gradients adds ~2.0 GB of transient activations, triggering an instant CUDA Out of Memory (OOM).

### 2. Sequence Length Sensitivity (Quadratic & Activation Scaling)
* Sequence length is the biggest lever on activation memory.
* For **8B models**:
  * `max_seq_length = 1024`: **~7.7 GB VRAM** (Safe).
  * `max_seq_length = 2048`: **~8.3 GB VRAM** (Will OOM unless `offload_embedding=True` and `torch.cuda.empty_cache()` are used).
  * `max_seq_length = 4096+`: Will exceed memory on 8B models.
* For **1B–3B models**: You can safely scale up to `4096` or `8192` tokens.

### 3. Laptop Thermal & Power Envelope (70W TDP)
* The RTX 5050 Laptop GPU runs on a 70W power limit.
* In our tests, sustained training stabilized at **~42W and 76°C**.
* **Recommendation for Long Training Runs (> 2 hours)**:
  * Elevate laptop base for airflow.
  * Avoid training while simultaneously running GPU-heavy desktop applications (browsers with hardware acceleration, video games) to prevent thermal throttling or VRAM preemption.

---

## 4. Hardware Optimization Rules for Your RTX 5050

When writing training scripts on this machine, always enforce these 5 golden rules:

### Rule 1: Always Explicitly Set `device_map="cuda"`
On 8GB single-GPU setups, Hugging Face `accelerate` calculates auto-split heuristics using unquantized weights and attempts to dispatch layers to CPU.
```python
model, tokenizer = FastModel.from_pretrained(
    model_name="unsloth/gemma-4-E4B-it",
    load_in_4bit=True,
    device_map="cuda",  # Prevents unnecessary CPU dispatch errors
)
```

### Rule 2: Keep Device Batch Size to 1 on 7B–8B Models
Scale effective batch size exclusively using gradient accumulation:
```python
per_device_train_batch_size = 1,
gradient_accumulation_steps = 8,  # Effective batch size = 8
```

### Rule 3: Use 8-bit Optimizers
Standard AdamW uses 32-bit floating point state for first and second moments (adding ~1.5 GB VRAM). Always specify:
```python
optim = "adamw_8bit"
```

### Rule 4: Use Unsloth Fast Patching & Gradient Checkpointing
Unsloth provides custom Triton kernels that rewrite gradient backprop with 0 additional memory overhead:
```python
use_gradient_checkpointing = "unsloth"
```

### Rule 5: Keep LoRA Rank Efficient ($r=8$ or $r=16$)
* $r=8, \alpha=8$: ~73 MB adapter size, lowest VRAM overhead, optimal for instruction tuning.
* $r=32+$: Increases activation gradients by ~400–600 MB. Only use $r \ge 32$ for smaller 1B–3B models where VRAM is plentiful.

---

## 5. Ideal Training Configurations by Use Case

### Configuration A: Gemma-4 E4B / Llama-3.1 8B (Maximum Capability Setup)
```python
max_seq_length = 1024
load_in_4bit = True

# LoRA
r = 8, lora_alpha = 8, lora_dropout = 0
finetune_vision_layers = False

# Trainer
per_device_train_batch_size = 1
gradient_accumulation_steps = 8
learning_rate = 2e-4
optim = "adamw_8bit"
```

### Configuration B: Llama-3.2 3B / Qwen-2.5 3B (High Speed & Long Context Setup)
```python
max_seq_length = 4096  # 4x longer context
load_in_4bit = True

# LoRA
r = 16, lora_alpha = 16, lora_dropout = 0

# Trainer
per_device_train_batch_size = 2
gradient_accumulation_steps = 4
learning_rate = 2e-4
optim = "adamw_8bit"
```
