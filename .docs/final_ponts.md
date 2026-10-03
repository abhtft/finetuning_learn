# Comprehensive Technical Guide: LLM Fine-Tuning Dynamics & Insights

A definitive reference documenting training telemetry analysis, mathematical foundations, and practitioner insights gathered from fine-tuning **Gemma-4 E4B (8.01B)** using **Unsloth (QLoRA 4-bit)** on consumer GPU hardware.

---

## 📑 Table of Contents
1. [Learning Rate & Schedulers: The 3-Stage Lifecycle](#1-learning-rate--schedulers-the-3-stage-lifecycle)
2. [Warm Starts Explained: Warmup, Checkpoints, and Servers](#2-warm-starts-explained-warmup-checkpoints-and-servers)
3. [Gradient Norm (`grad_norm`): Mechanics & Diagnostic Health](#3-gradient-norm-grad_norm-mechanics--diagnostic-health)
4. [Loss Dynamics: Epoch 1 vs. Epoch 2 Learning Distribution](#4-loss-dynamics-epoch-1-vs-epoch-2-learning-distribution)
5. [Understanding Loss Jitter: Why Loss Oscillates Up & Down](#5-understanding-loss-jitter-why-loss-oscillates-up--down)
6. [Critical Insights & Architectural Best Practices (Beyond the Basics)](#6-critical-insights--architectural-best-practices-beyond-the-basics)
   - 6.1 Effective Batch Size & Gradient Accumulation
   - 6.2 Response-Only Loss Masking (`labels = -100`)
   - 6.3 LoRA Parameter Scaling: Rank $r$, Alpha $\alpha$, and RSLoRA
   - 6.4 Memory Optimization: QLoRA (NF4) and 8-bit AdamW
   - 6.5 Evaluation Strategy: Early Stopping & Train-Val Splits
   - 6.6 Post-Training Inference Calibration & Repetition Penalties

---

## 1. Learning Rate & Schedulers: The 3-Stage Lifecycle

### What is Learning Rate (LR)?
While gradients define the **direction and steepness** of the loss surface, the **Learning Rate ($\eta$)** is the step-size multiplier that determines how far parameters move on each optimization step:

$$\Delta W = -\eta \cdot \frac{m_t}{\sqrt{v_t} + \epsilon}$$

*(where $m_t$ and $v_t$ are the first and second moment estimates of AdamW).*

```
Learning Rate
  ▲
  │        (Peak ~1.8e-4 @ Step 6-10)
  │            ████
  │          ██    ██
  │        ██        ██
  │       █            ██
  │      █               ██
  │     █                  ██  (Step 50 / Epoch 1: ~1.02e-4)
  │    █                     ██
  │   █                        ██
  │  █                           ██
  │ █                              ██
  │█                                 ██
  │                                    ██ (Step 100: ~5.03e-8)
  └────────────────────────────────────────► Global Steps (100)
    ◄────► ◄──────────────────────────────►
    Warmup              Cosine Decay
    (1-6)                 (7-100)
```

### The 3 Stages of Cosine Annealing with Warmup:
1. **Stage 1: Linear Warmup (Steps 1 – 6 | LR: $0.0 \rightarrow 1.8\times 10^{-4}$):**
   - Ramps up learning rate linearly from zero to peak.
   - Prevents destabilizing newly added adapter layers before AdamW accumulates valid moving averages of gradients.
2. **Stage 2: Peak Learning Phase (Steps 6 – 20 | LR: $\approx 1.8\times 10^{-4}$):**
   - Maximum optimization capacity. The model rapidly absorbs prompt-response templates and instruction-following conventions.
3. **Stage 3: Cosine Annealing / Decay (Steps 20 – 100 | LR: $1.8\times 10^{-4} \rightarrow 5.03\times 10^{-8}$):**
   - Progressively reduces step size following a half-cosine wave.
   - Allows fine parameter updates, helping the model settle into the deepest point of the local loss basin without bouncing across steep walls.

### Why $1.8\times 10^{-4}$ for LoRA vs. Full Fine-Tuning?
* **Full Fine-Tuning ($1\times 10^{-5} - 2\times 10^{-5}$):** Updates all 8+ billion parameters; higher rates cause catastrophic forgetting and weight disruption.
* **LoRA Fine-Tuning ($1\times 10^{-4} - 3\times 10^{-4}$):** Only trains low-rank adapter matrices ($\approx 0.1\%$ of parameters). Requires an order-of-magnitude larger learning rate to achieve significant representation shift in the adapter subspace.

---

## 2. Warm Starts Explained: Warmup, Checkpoints, and Servers

In modern LLM engineering, the term **"Warm Start"** appears in three distinct contexts:

### A. Learning Rate Warmup
* **Definition:** Gradually increasing learning rate from 0 to peak over the initial $N$ steps (`warmup_ratio=0.06` $\rightarrow$ 6 steps).
* **Observed Data:** Step 5 was at `lr = 0.000120`, reaching peak `lr = 0.0001795` at Step 10 before decaying.

### B. Checkpoint / Model Warm Start (vs. Cold Start)
* **Cold Start:** Training a model from randomly initialized Gaussian weights with zero prior knowledge.
* **Warm Start:** Initializing training from a pre-trained base model (`unsloth/gemma-4-E4B-it`) or resuming training from an intermediate checkpoint (`gemma_4_e4b_output/checkpoint-100`) to retain previously learned weights, optimizer moments, and scheduler states:
  ```python
  trainer.train(resume_from_checkpoint="gemma_4_e4b_output/checkpoint-100")
  ```

### C. Inference / Server Warm Start
* **Definition:** Pre-loading model weights, compiling CUDA kernels, and initializing KV cache structures in GPU memory.
* **Benefit:** Subsequent user queries execute with sub-millisecond response latency, bypassing model load and graph compilation overhead.

---

## 3. Gradient Norm (`grad_norm`): Mechanics & Diagnostic Health

### Mathematical Definition
During backpropagation, the loss function computes partial derivatives for all trainable LoRA parameters. The **Gradient Norm** is the Euclidean ($L_2$) length of this concatenated gradient vector $g = \nabla_\theta \mathcal{L}$:

$$\text{grad\_norm} = \|g\|_2 = \sqrt{\sum_{i=1}^{P} g_i^2}$$

### The Mountain Analogy (Loss vs. Gradient Norm)
* **Loss = Altitude:** Measures how high up you are from the lowest valley (error magnitude).
* **Gradient Norm = Slope Steepness:** Measures how steep the ground is under your feet (force and magnitude of weight update).

```
High Loss, High Grad Norm ────────► [Steep cliff high on mountain] (Early adaptation)
Low Loss, Low Grad Norm   ────────► [Flat valley floor] (Optimal convergence)
High Loss, Zero Grad Norm ────────► [Trapped on a flat plateau] (Vanishing gradient / stalled learning)
```

### Telemetry Health Spectrum
| Metric Behavior | Meaning | Status in Your Gemma-4 Run |
| :--- | :--- | :--- |
| **Early Steps (1–15)** | High values ($0.83 - 0.89$) as LoRA adapts to turn tags. | ✅ Normal & Expected |
| **Middle & Late Steps (20–100)** | Settles into tight corridor ($0.19 - 0.29$). | ✅ Flawless stability |
| **Exploding Gradient ($>10.0$)** | Dangerous instability; causes NaN loss. | ❌ Zero occurrences |
| **Vanishing Gradient ($<10^{-6}$)** | Optimizer stalled; loss stops decreasing. | ❌ Zero occurrences |
| **Gradient Clipping (`max_grad_norm=1.0`)** | Truncates gradients exceeding threshold. | 🛡️ Gradients never exceeded 0.90 |

---

## 4. Loss Dynamics: Epoch 1 vs. Epoch 2 Learning Distribution

### Quantitative Breakdown
Across the 100-step run, the learning was disproportionately concentrated in the first epoch:

```
Loss
1.7│  █ (Step 5: 1.6543)
1.5│  ██
1.3│  │ \
1.1│  │  \   EPOCH 1: 89% of Total Learning (Loss: 1.6543 ➔ 0.7647)
0.9│  │   \  [Format, chat template, turn boundaries mastered]
0.8│  │    \───────\
0.7│  │             \───► (Step 50: 0.7647)
0.7│  │                    \────── EPOCH 2: 11% Consolidation (Loss: 0.7647 ➔ 0.6550)
0.6│  │                           \────► (Step 100: 0.6550)
   └──┴──────────────────────────────────────────────────────► Global Steps
      0                     50 (Epoch 1)            100 (Epoch 2)
```

### Key Takeaway for Practitioners:
* **Epoch 1:** Mastered conversational roles (`<start_of_turn>user`, `<start_of_turn>model`), tone, and structural constraints.
* **Epoch 2:** Performed minor weight consolidation with small learning rates ($10^{-5} \to 10^{-8}$).
* **Rule of Thumb:** For standard instruction fine-tuning on high-quality datasets, **1 Epoch is generally sufficient**. Training for 1 epoch achieves ~95% of target model capability while cutting wall-clock compute time and thermal wear by **50%**.

---

## 5. Understanding Loss Jitter: Why Loss Oscillates Up & Down

During training, loss values exhibit micro-fluctuations (e.g., Step 70: `0.6896` $\rightarrow$ Step 85: `0.7818` $\rightarrow$ Step 90: `0.6788`). This is driven by 4 factors:

### 1. Batch Heterogeneity (Sample Difficulty Variance)
* Each logging interval averages loss across only **40 samples** (5 steps $\times$ 8 accumulation steps).
* **Easy Batches (Loss $\sim 0.65$):** Simple factual questions, short definitions.
* **Hard Batches (Loss $\sim 0.78$):** Complex multi-step reasoning, mathematical derivations, or dense code blocks.
* A temporary loss rise indicates a more challenging batch, not model degradation.

### 2. Stochastic Mini-Batch Sampling vs. Full Batch
* We calculate gradients on mini-batches, not all 400 dataset samples simultaneously.
* Mini-batch optimization introduces natural stochastic variance around the macro downward path.

### 3. Token-Level Cross-Entropy Averaging
* Loss is computed as negative log-likelihood per token: $\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \log P(w_i | w_{<i})$.
* Dense reasoning samples with rare vocabulary produce naturally higher token entropy.

### 4. Jitter as a Health Indicator
* **Healthy Sign:** Downward moving corridor with small oscillations ($\pm 0.05 - 0.10$).
* **Overfitting Warning:** A perfectly monotonic, jitter-free drop to near-zero ($<0.05$) indicates exact memorization of training text rather than generalized reasoning.

---

## 6. Critical Insights & Architectural Best Practices (Beyond the Basics)

### 6.1 Effective Batch Size & Gradient Accumulation
Consumer GPUs cannot fit large batch sizes into VRAM for an 8B model. Gradient accumulation bridges this constraint:

$$\text{Effective Batch Size} = \text{per\_device\_batch\_size} \times \text{gradient\_accumulation\_steps} \times \text{num\_gpus}$$

$$\text{In your run: } 1 \times 8 \times 1 = 8\text{ samples per optimizer step}$$

* **Why not batch size 1 without accumulation?** Small batch sizes yield noisy gradient estimates that cause erratic parameter trajectory. Accumulating across 8 micro-batches averages out outlier noise before taking an AdamW step.

---

### 6.2 Response-Only Loss Masking (`train_on_responses_only`)
When fine-tuning instruction models, prompt tokens must **NOT** contribute to the loss:

```
[<start_of_turn>user\nExplain gravity.<end_of_turn>\n<start_of_turn>model\n] [Gravity is a fundamental force...]
◄────────────────────── PROMPT (MASKED: labels = -100) ───────────────────► ◄── RESPONSE (TRAINED: loss active) ──►
```

* **Mechanism:** PyTorch's `CrossEntropyLoss(ignore_index=-100)` ignores all tokens labeled with `-100`.
* **Impact:** Prevents the model from wasting parameter capacity learning to predict the user's prompt text, focusing 100% of gradient updates on generating high-quality assistant responses.

---

### 6.3 LoRA Parameter Scaling: Rank $r$, Alpha $\alpha$, and RSLoRA
In LoRA, parameter updates are parameterized as $\Delta W = \frac{\alpha}{r} (B \cdot A)$:

* **Rank ($r$):** The inner dimension of the low-rank decomposition.
  * $r = 8$: Great for style, format, and tone adaptation.
  * $r = 16 - 32$: Recommended for complex reasoning, code generation, and domain-specific knowledge acquisition.
* **Alpha ($\alpha$):** The scaling factor for weight updates.
  * Standard convention: Set $\alpha = r$ (multiplier = $1.0$) or $\alpha = 2r$ (multiplier = $2.0$).
* **RSLoRA (Rank-Stabilized LoRA):**
  * Uses scaling $\frac{\alpha}{\sqrt{r}}$ instead of $\frac{\alpha}{r}$.
  * Eliminates vanishing/exploding adapter dynamics when scaling to larger ranks ($r \ge 32$).

---

### 6.4 Memory Optimization: QLoRA (NF4) and 8-bit AdamW
Fine-tuning an 8B model on an 8GB VRAM envelope requires two key memory savings:

1. **NF4 (Normal Float 4) Quantization:**
   * Quantizes frozen base model weights from 16-bit to 4-bit (saving ~75% model memory: 16 GB $\rightarrow$ 4.5 GB).
   * LoRA adapters are kept in full 16-bit precision (`bfloat16`) for accurate gradient flow.
2. **8-bit AdamW (`optim="adamw_8bit"`):**
   * Standard 32-bit AdamW maintains two FP32 states ($m_t, v_t$) per trainable parameter (8 bytes/param).
   * 8-bit AdamW compresses optimizer states to 2 bytes/param, saving 75% optimizer VRAM without accuracy loss.

---

### 6.5 Evaluation Strategy: Early Stopping & Train-Val Splits
For future production-grade runs, adhere to the standard validation protocol:

```python
# Recommended Production Split & Evaluation Config:
dataset_split = dataset.train_test_split(test_size=0.10, seed=3407)

training_args = SFTConfig(
    eval_dataset=dataset_split["test"],
    eval_strategy="steps",
    eval_steps=20,                    # Evaluate every 20 steps
    save_strategy="steps",
    save_steps=20,
    load_best_model_at_end=True,      # Automatically restore best checkpoint
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)
```

* **Validation Loss vs. Training Loss:**
  * If `train_loss` drops while `eval_loss` rises $\rightarrow$ **Overfitting has begun; stop training.**
  * If both decrease in parallel $\rightarrow$ **True generalization.**

---

### 6.6 Post-Training Inference Calibration & Repetition Penalties
When deploying the trained adapter ([infer_chat.py](file:///home/abhishek/projects/unsloth/infer_chat.py)):

| Parameter | Recommended Value | Technical Purpose |
| :--- | :--- | :--- |
| `temperature` | $0.6 - 0.7$ | Balances creativity with deterministic instruction compliance. |
| `top_p` | $0.90 - 0.95$ | Nucleus sampling; filters low-probability tail tokens. |
| `repetition_penalty` | $1.02 - 1.05$ | Prevents cyclic loops while avoiding unnatural indentation/tab suppression (values $> 1.15$ can suppress syntax formatting). |
| `max_new_tokens` | $512 - 1024$ | Sets hard boundary for output length. |

---

## 7. Summary Reference Card

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              Fine-Tuning Diagnostic Matrix                             │
├─────────────────────┬──────────────────────────┬───────────────────────────────────────┤
│ Observed Metric     │ Healthy Target Range     │ What to Do If Outside Range           │
├─────────────────────┼──────────────────────────┼───────────────────────────────────────┤
│ Final Loss          │ 0.50 – 0.75              │ <0.20: Overfitting; >1.0: Increase LR │
│ Gradient Norm       │ 0.10 – 0.50 (post-warmup)│ >2.0: Lower LR or enable grad clip    │
│ Warmup Fraction     │ 3% – 10% of total steps  │ 0%: High risk of early NaN gradient   │
│ Epoch Count         │ 1 – 2 full epochs        │ >3 epochs: Check validation loss      │
│ Loss Reduction      │ 50% – 70% from baseline  │ <20%: Check template/data masking     │
└─────────────────────┴──────────────────────────┴───────────────────────────────────────┘
```
