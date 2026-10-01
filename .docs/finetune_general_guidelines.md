# LLM Fine-Tuning & Best Practices Guide

A comprehensive guide covering concepts, hardware optimization, hyperparameter tuning, dataset preparation, and troubleshooting for fine-tuning Large Language Models (LLMs) with **Unsloth**, **LoRA/QLoRA**, and **Hugging Face TRL**.

---

## 1. Fine-Tuning Paradigms

| Method | VRAM Requirement | Trainable Parameters | Best Use Case |
| :--- | :--- | :--- | :--- |
| **Full Fine-Tuning** | Extreme (16–80GB+ per GPU) | 100% of weights | Pre-training, foundation model domain shifts |
| **LoRA (16-bit)** | Moderate (12–24GB) | ~0.1% – 2% | Instruction tuning, task adaptation |
| **QLoRA (4-bit)** | Minimal (6–12GB) | ~0.1% – 2% | Consumer GPUs (e.g. 8GB RTX 30/40/50 series) |

---

## 2. Hardware & Memory (VRAM) Optimization

When training on consumer hardware (e.g., 8GB – 16GB VRAM):

### A. Quantization
* **4-bit (`load_in_4bit=True`)**: Uses BitsAndBytes NF4 (NormalFloat4) quantization to compress 16-bit model weights down to 4-bit while maintaining precision in compute layers. Cuts base model weight memory by ~75%.
* **Explicit `device_map`**: On single-GPU setups with tight memory, specify `device_map="cuda"` or `device_map={"": 0}` in `FastModel.from_pretrained` to prevent Accelerate from prematurely offloading layers to CPU.

### B. Effective Batch Size Management
$$ \text{Effective Batch Size} = \text{per\_device\_train\_batch\_size} \times \text{gradient\_accumulation\_steps} \times \text{number\_of\_gpus} $$

* **Target Effective Batch Size**: Typically **8 to 64** for instruction tuning.
* **Low VRAM Rule**: Keep `per_device_train_batch_size = 1` or `2`, and increase `gradient_accumulation_steps` (e.g., `4`, `8`, or `16`) to maintain smooth gradients without OOM.

### C. Sequence Length (`max_seq_length`)
* Attention memory scales quadratically or linearly with sequence length.
* Measure your dataset's 95th-percentile token length:
  * Short Q&A / Instruction: `512` – `1024` tokens (fits easily in 8GB VRAM).
  * Long document / Reasoning / Chain-of-Thought: `2048` – `4096` tokens.

### D. Memory-Saving Techniques
* **8-bit Optimizers (`optim="adamw_8bit"`)**: Cuts optimizer state memory by ~75% compared to standard 32-bit AdamW.
* **Gradient Checkpointing (`use_gradient_checkpointing="unsloth"`)**: Recomputes activations during the backward pass instead of storing them all in VRAM. Unsloth provides optimized custom kernels for this.

---

## 3. LoRA & QLoRA Hyperparameters

### A. Rank ($r$) and Alpha ($\alpha$)
* **Rank ($r$)**: Represents the dimensional width of the low-rank update matrices.
  * **$r = 8$ or $16$**: Optimal for standard instruction following, classification, style transfer.
  * **$r = 32$ or $64$**: Suitable for complex reasoning, mathematical logic, code generation, or learning new knowledge.
* **Alpha ($\alpha$)**: Scaling multiplier for the adapter weights ($\text{weight update} = \frac{\alpha}{r} \times \Delta W$).
  * **Rule of Thumb**: Set $\alpha = r$ (multiplier 1.0) or $\alpha = 2 \times r$ (multiplier 2.0).

### B. Target Modules
* Fine-tuning **all linear modules** (Self-Attention + MLP) yields the best quality and generalization:
  * **Attention Modules**: `q_proj`, `k_proj`, `v_proj`, `o_proj` (Controls attention context and retrieval).
  * **MLP Modules**: `gate_proj`, `up_proj`, `down_proj` (Stores factual knowledge and reasoning patterns).
* For multimodal models, disable vision adapter tuning (`finetune_vision_layers=False`) if you are only adapting text capabilities.

### C. LoRA Dropout
* Set `lora_dropout = 0` when using Unsloth to utilize optimized fast Triton kernels.
* If training small datasets prone to extreme overfitting, use standard LoRA with `0.05` – `0.1`.

---

## 4. Dataset Preparation & Quality Rules

> **Data Quality > Data Quantity**: 1,000 carefully curated, high-quality, diverse instruction-response pairs often outperform 100,000 noisy scraped samples.

### A. Chat Templates & Formatting
* Always format multi-turn conversations using the model's official chat template (`tokenizer.apply_chat_template(...)` or Unsloth's `get_chat_template`).
* Avoid duplicate BOS/EOS tokens (`<bos>`, `<s>`, `<end_of_turn>`). Strip manual prefix BOS if the template already injects it.

### B. Train on Responses Only (Prompt Loss Masking)
* **Standard Fine-Tuning**: Computes loss across both user questions and assistant responses. The model wastes capacity learning to predict the user prompt.
* **Response-Only Fine-Tuning (`train_on_responses_only`)**: Masks user and system tokens (sets label to `-100`). The model is only penalized for errors in the assistant's output, improving response quality and reducing hallucinations.

---

## 5. Training Dynamics & Hyperparameter Guidelines

| Parameter | Recommended Value | Notes |
| :--- | :--- | :--- |
| **Learning Rate** | `1e-4` to `2e-4` (LoRA/QLoRA)<br>`1e-5` to `5e-5` (Full) | Lower for larger models (e.g. 70B: `5e-5`). |
| **LR Scheduler** | `linear` or `cosine` | Cosine provides smoother convergence towards the end. |
| **Warmup Ratio** | `0.03` to `0.06` (3%–6% of total steps) | Prevents large initial gradient spikes from disrupting pretrained weights. |
| **Weight Decay** | `0.001` to `0.01` | Helps regularize and avoid exploding adapter weights. |
| **Epochs** | `1` to `3` epochs | LLMs overfit quickly. 1–2 epochs are usually sufficient for instruction tuning. |
| **Seed** | Fixed (e.g. `3407`) | For deterministic reproducibility. |

---

## 6. Training Checklist & Best Practices

1. **Baseline Before Fine-Tuning**: Benchmark base model zero-shot / few-shot performance on your validation set before running training.
2. **Loss Curve Monitoring**:
   * Initial loss typically starts around `1.8` – `2.5` (depending on tokenizer/task).
   * Smooth, steady decrease to `0.8` – `1.2`.
   * If loss drops abruptly to `< 0.2` within a few steps, check for data leakage, repetitive dataset rows, or template mismatch.
3. **Evaluate Mid-Training**:
   * Use an evaluation dataset (`eval_dataset` in `SFTTrainer`) with `eval_steps` or `evaluation_strategy="steps"`.
   * Watch for validation loss divergence (a clear indicator of overfitting).

---

## 7. Saving, Merging & Inference

### A. Saving LoRA Adapters (Lightweight)
```python
model.save_pretrained("my_lora_adapters")
tokenizer.save_pretrained("my_lora_adapters")
```
Saves only the modified low-rank weights (~20MB to ~150MB).

### B. Merging Weights for Fast Inference (vLLM / Ollama / GGUF)
Unsloth allows exporting to multiple deployment formats directly:

* **Save 16-bit Merged Model**:
  ```python
  model.save_pretrained_merged("merged_model_16bit", tokenizer, save_method="merged_16bit")
  ```
* **Save Quantized 4-bit Merged Model**:
  ```python
  model.save_pretrained_merged("merged_model_4bit", tokenizer, save_method="merged_4bit")
  ```
* **Export to GGUF (for Ollama / llama.cpp)**:
  ```python
  model.save_pretrained_gguf("gemma_model_gguf", tokenizer, quantization_method="q4_k_m")
  ```

---

## 8. Troubleshooting Common Errors

### 1. `ValueError: Some modules are dispatched on the CPU or the disk`
* **Cause**: Accelerate auto-device heuristic underestimates available VRAM for 4-bit quantization and tries to offload layers to CPU.
* **Fix**: Pass `device_map="cuda"` or `device_map={"": 0}` explicitly inside `FastModel.from_pretrained(...)`.

### 2. CUDA Out of Memory (OOM)
* **Checklist**:
  1. Reduce `max_seq_length` (e.g., from 2048 to 1024 or 512).
  2. Set `per_device_train_batch_size = 1`.
  3. Increase `gradient_accumulation_steps` to compensate.
  4. Ensure `load_in_4bit = True` and `optim = "adamw_8bit"`.
  5. If multimodal, set `finetune_vision_layers = False`.

### 3. Loss is `0.0` or Tokens are Ignored
* **Cause**: Chat template instruction delimiter mismatch inside `train_on_responses_only`.
* **Fix**: Ensure the dataset conversations use standard roles (`user`, `assistant`) and use `get_chat_template(tokenizer, chat_template="...")` matching your model family.

### 4. Generation Loops / Repetitive Responses
* **Cause**: Missing End-Of-Sequence (`<eos>` or `<end_of_turn>`) token during training or over-training (too many epochs).
* **Fix**: Verify chat template appends `<end_of_turn>` / `<eos>` at the end of assistant turns, and lower training epochs to `1` or `2`.


