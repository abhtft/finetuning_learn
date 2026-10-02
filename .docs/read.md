Here is a curated reading list of the most foundational papers and practical documentation directly related to the fine-tuning stack used in this repository (Unsloth, LoRA / QLoRA, and Hugging Face TRL).

📚 1. Must-Read Core Papers (Read in this Order)
#	Paper Title	Authors / Year	Why It's Essential for This Repo
1	LoRA: Low-Rank Adaptation of Large Language Models	Edward J. Hu et al. (2021)	The foundation of modern PEFT. Explains why freezing base weights and training tiny rank decomposition matrices ($A$ and $B$) achieves equal/better performance than full fine-tuning with 10,000× fewer trainable parameters.
2	QLoRA: Efficient Finetuning of Quantized LLMs	Tim Dettmers et al. (2023)	The reason you can fine-tune on an 8GB GPU. Introduces 4-bit NormalFloat (NF4) quantization, double quantization, and paged optimizers to reduce base model memory by ~75%.
3	LIMA: Less Is More for Alignment	Chunting Zhou et al. (2023)	Why dataset quality beats dataset quantity. Proves that fine-tuning on just 1,000 clean, curated instruction samples can produce better conversational models than training on 50,000+ noisy rows.
4	Direct Preference Optimization (DPO)	Rafael Rafailov et al. (2023)	Post-training alignment. Replaces complex RLHF (PPO + separate reward model) with a simple binary cross-entropy loss over chosen vs. rejected responses.
📖 2. Best Practical Guides & Documentation
A. Unsloth Technical Guides
Unsloth Official Documentation: Explains how custom Triton kernels, manual backward passes, and memory optimizations achieve 2× faster training and 70% less VRAM usage.
Unsloth Gradient Accumulation & Loss Smoothing: A concise post detailing subtle mathematical bugs in standard gradient accumulation and how Unsloth fixes them.
B. Hugging Face Guides
Hugging Face TRL (Transformer Reinforcement Learning) Docs: Practical tutorial on SFTTrainer (Supervised Fine-Tuning Trainer), formatting chat templates, and masking prompt tokens during loss calculation.
Hugging Face PEFT Conceptual Guide: Visual explanation of parameter-efficient methods, target projection layers (q_proj, v_proj, gate_proj, etc.), and adapter merging.
📂 3. Local Guides in This Repository
Your repository already contains customized cheat sheets and hardware profiling:



.docs/finetune_general_guidelines.md

Detailed breakdown of hyperparameters ($r$, $\alpha$, learning rate schedules, optimizer selection).
Dataset formatting best practices and loss curve diagnostics.


.docs/hardware_guidelines.md

Real benchmark data for your RTX 5050 8GB GPU running Gemma-4.
VRAM budget math and maximum sequence lengths before encountering Out-of-Memory (OOM).
🚀 Recommended 3-Step Reading Plan
Step 1 (Start Here): Read sections 1–3 of the LoRA Paper to understand the concept of low-rank matrices ($W + \Delta W$).
Step 2: Read section 3 of the QLoRA Paper to see how 4-bit NF4 compression works without sacrificing precision.
Step 3: Review 

.docs/finetune_general_guidelines.md
 to apply these concepts directly to 

train_gemma4_e4b.py
.