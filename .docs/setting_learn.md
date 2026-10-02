# Systematic Hyperparameter Experimentation & Tuning Guide

A definitive guide on how AI practitioners scientifically explore, isolate, benchmark, and optimize LLM fine-tuning configurations. 

This document decodes the **empirical experiment journey**, explains the mathematical intuition behind every parameter shift (Rank, RSLoRA, Learning Rate, Epochs), and outlines the **gold-standard methodology** for running ablation studies.

---

## 1. Deconstructing the 8-Experiment Journey

The table below illustrates a classic, disciplined empirical ablation study:

```
┌─────┬────────────┬─────────┬─────────┬────────┬────────────┬────────────────────────────────────────────────────────┐
│ Exp │  LoRA r    │ Epochs  │ Samples │   LR   │ Train Loss │ Key Finding / Empirical Insight                        │
├─────┼────────────┼─────────┼─────────┼────────┼────────────┼────────────────────────────────────────────────────────┤
│ 01  │ 16         │ 0.13    │ 3k      │ 2e-4   │ 2.916      │ Baseline: Undertrained, high loss                      │
│ 02  │ 32         │ 0.24    │ 5k      │ 2e-4   │ 1.725      │ Higher rank adds capacity for complex patterns         │
│ 03  │ 64+RSLoRA  │ 0.20    │ 10k     │ 2e-4   │ 1.460      │ RSLoRA stabilizes higher rank updates                  │
│ 04  │ 64+RSLoRA  │ 0.40    │ 20k     │ 1e-4   │ ~1.05      │ Lower LR improves gradient stability on larger data    │
│ 05  │ 128+RSLoRA │ 0.40    │ 20k     │ 5e-5   │ 1.134      │ Diminishing returns: r=128 learns slower than r=64     │
│ 06  │ 64+RSLoRA  │ 3.00    │ 10k     │ 1e-4   │ ~0.30      │ Multi-epoch allows complete weight convergence         │
│ 07  │ 128+RSLoRA │ 3.00    │ 10k     │ 1e-4   │ ~0.59      │ r=64 generalizes better & converges faster than r=128  │
│ 08  │ 64+RSLoRA  │ 5.00    │ 10k     │ 7e-5   │ 0.0115     │ 5 epochs leads to near-zero loss (memorization alert)  │
└─────┴────────────┴─────────┴─────────┴────────┴────────────┴────────────────────────────────────────────────────────┘
```

---

## 2. Core Concepts: Why Each Setting Behave Like This

### A. LoRA Rank ($r$) & The Law of Diminishing Returns
* **Rank ($r$)** controls the bottleneck width of the low-rank update matrices $\Delta W = B \times A$.
* **Low Rank ($r=8, 16$):** Excellent for shallow style transfer or simple classification. Lacks expressiveness for complex reasoning or multi-turn conversational nuance.
* **Medium-High Rank ($r=32, 64$):** Captures multi-step reasoning, mathematical logic, and complex syntax.
* **Excessive Rank ($r=128+$):** Doubles parameter count and memory with **diminishing returns**. As seen in *Exp 05 vs Exp 04*, higher ranks require more data and slower learning rates to converge, often performing worse than $r=64$.

---

### B. Standard LoRA vs. RSLoRA (Rank-Stabilized LoRA)

In standard LoRA, the adapter weight update is scaled by $\frac{\alpha}{r}$:
$$\Delta W_{\text{standard}} = \frac{\alpha}{r} (B \cdot A)$$

* **The Problem with Standard LoRA at High Ranks ($r \ge 64$):** 
  As rank $r$ increases, dividing by $r$ collapses the magnitude of the gradient updates, making high-rank adapters learn progressively slower.

* **The RSLoRA Solution:** 
  Rank-Stabilized LoRA scales by $\frac{\alpha}{\sqrt{r}}$:
  $$\Delta W_{\text{RSLoRA}} = \frac{\alpha}{\sqrt{r}} (B \cdot A)$$
  This maintains consistent gradient magnitude regardless of rank, unlocking the true expressive power of $r=64$ and $r=128$.

In Unsloth / PEFT, RSLoRA is enabled with:
```python
model = FastModel.get_peft_model(
    model,
    r=64,
    lora_alpha=16,
    use_rslora=True, # Enables gamma = alpha / sqrt(r) scaling
)
```

---

### C. Fractional Epochs ($0.13 - 0.40$) vs. Full Multi-Epochs ($1 - 3$)
* **Fractional Epochs ($< 1.0$):** In experiments 01–05, the model only saw $13\%$ to $40\%$ of the dataset. The model only learned superficial vocabulary associations without consolidating internal reasoning representations.
* **Multi-Epoch ($3$ Epochs):** Passing over high-quality data 3 times (*Exp 06*) dropped loss drastically from $1.05 \rightarrow 0.30$.
* **Over-fitting / Memorization Warning (*Exp 08* - 5 Epochs, Loss 0.0115):** 
  When training loss approaches $\approx 0.01$, the model has essentially memorized the training dataset verbatim, leading to catastrophic degradation on unseen test inputs.

---

## 3. The 5-Phase Scientific Tuning Framework

Never change multiple variables at once without a baseline. Follow this standard 5-phase experimentation loop:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        Scientific Hyperparameter Exploration Loop                      │
└────────────────────────────────────────────────────────────────────────────────────────┘

   Phase 1: Sanity Check ────► 30-50 steps to verify VRAM, throughput & loss descent
             │
   Phase 2: Capacity     ────► Sweep LoRA Rank (r = 16 vs 32 vs 64 + RSLoRA)
             │
   Phase 3: Learning Rate────► Sweep LR (2e-4 vs 1e-4 vs 5e-5) with Cosine Schedule
             │
   Phase 4: Epoch Scale  ────► Train for 1, 2, and 3 full epochs; track Eval Loss
             │
   Phase 5: Regularize   ────► Tune Weight Decay (0.01) & Dropout to prevent memorization
```

---

### Phase 1: Smoke Test & Sanity Check
* **Goal:** Verify pipeline integrity, memory envelope, and CUDA stability.
* **Settings:** `max_steps = 30`, `batch_size = 1`, `grad_accum = 4`, `lr = 2e-4`.
* **Success Metric:** No CUDA OOM, loss decreases by at least 30–50% from step 1 to 30.

---

### Phase 2: Adapter Capacity & Rank Optimization
* Keep data size and learning rate constant (e.g. 5,000 samples, $\text{LR} = 1.5\text{e-}4$).
* **Experiment Progression:**
  1. $r = 16, \alpha = 16$ (Baseline)
  2. $r = 32, \alpha = 32$ (Assess capacity gain)
  3. $r = 64, \alpha = 16, \text{use\_rslora=True}$ (High-capacity stabilized)
* **Metric:** Pick the lowest rank that gives significant loss reduction without slowing down step latency.

---

### Phase 3: Learning Rate & Schedule Calibration
As rank increases or batch size scales, adjust the learning rate:

$$\text{Ideal LR} \propto \frac{1}{\sqrt{\text{LoRA Rank}}} \quad \text{and} \quad \text{Ideal LR} \propto \sqrt{\text{Effective Batch Size}}$$

| Scenario | Recommended Learning Rate | Schedule |
| :--- | :--- | :--- |
| **Standard QLoRA ($r=8, 16$)** | $2.0 \times 10^{-4}$ | Cosine (5% Warmup) |
| **High-Rank RSLoRA ($r=64$)** | $1.0 \times 10^{-4}$ | Cosine (6% Warmup) |
| **High-Rank RSLoRA ($r=128$)** | $5.0 \times 10^{-5} - 7.0 \times 10^{-5}$ | Cosine (10% Warmup) |

---

### Phase 4: Full Multi-Epoch Convergence vs. Overfitting
Always split your data: **90% Training / 10% Validation**.

```
Loss
 ▲
 │   \
 │    \  Training Loss (Continues dropping)
 │     \──────\
 │      \      \────────\
 │       \   Eval Loss   \───────  Optimal Stopping Point (Epoch 2 - 3)
 │        \────────────/─────────  Overfitting Zone: Eval loss rises (Epoch 5+)
 └────────────────────────────────────────► Epochs
```

* **Underfitting:** Training Loss $> 1.0$, Eval Loss $> 1.0$. $\rightarrow$ *Increase epochs or increase rank.*
* **Sweet Spot:** Training Loss $\approx 0.30 - 0.60$, Eval Loss $\approx 0.35 - 0.65$. $\rightarrow$ *Best generalization.*
* **Overfitting:** Training Loss $\le 0.05$, Eval Loss $\ge 0.80$. $\rightarrow$ *Model has memorized strings; reduce epochs.*

---

## 4. Practical Ablation Tracking Template

Use this markdown table or integrate **Weights & Biases / MLflow** to log your own runs:

```markdown
| Exp ID | LoRA Rank | Scaling Mode | Samples | Epochs | Batch (Eff) | Peak LR | Final Train Loss | Final Eval Loss | Subjective Output Quality |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Run-01 | r=16      | Standard     | 1,000   | 1.0    | 8           | 2.0e-4  | 1.42             | 1.55            | Coherent, minor repetition|
| Run-02 | r=32      | Standard     | 1,000   | 1.0    | 8           | 1.8e-4  | 0.98             | 1.08            | Clear improvement in tone |
| Run-03 | r=64      | RSLoRA       | 5,000   | 2.0    | 16          | 1.0e-4  | 0.42             | 0.48            | Strong logic & no glitches|
| Run-04 | r=64      | RSLoRA       | 5,000   | 4.0    | 16          | 1.0e-4  | 0.08             | 0.72            | Overfitted; repeats data  |
```

---

## 5. Golden Rules for Setting Tuning

1. **Change One Axis at a Time:** If you change rank and learning rate and dataset size simultaneously, you cannot know which change caused the improvement or failure.
2. **Never Rely on Training Loss Alone:** A train loss of $0.01$ looks impressive but often produces broken, brittle models. Always track **Eval Loss**.
3. **Data Quality > Hyperparameter Tweaks:** Cleaning 5,000 dirty rows will improve your model far more than jumping from $r=32$ to $r=128$.
4. **Use RSLoRA for Ranks $\ge 32$:** Avoid gradient decay at higher ranks by enabling `use_rslora=True`.
5. **Target the Sweet Spot:** For conversational instruction tuning, aim for **$r=32$ to $64$**, **$1 - 3$ epochs**, and an effective batch size of **$8 - 16$** with **cosine LR decay**.
