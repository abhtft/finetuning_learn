import math
import torch
from datasets import load_dataset
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template, standardize_data_formats, train_on_responses_only
from trl import SFTTrainer, SFTConfig

# ==============================================================================
# 1. Configuration & Model Loading
# ==============================================================================
MODEL_PATH = "gemma_4_e4b_lora"  # Path to saved fine-tuned LoRA adapters
MAX_SEQ_LENGTH = 1024
EVAL_SAMPLE_START = 400          # Start index of held-out validation samples
EVAL_SAMPLE_END = 800            # 400 unseen samples (400 -> 800)
FINAL_TRAIN_LOSS = 0.6550        # Training loss from step 100

print("=" * 70)
print(f"🔍 EVALUATION & OVERFITTING DIAGNOSIS: {MODEL_PATH}")
print("=" * 70)

print(f"\n📦 Loading model and tokenizer from '{MODEL_PATH}'...")
model, tokenizer = FastModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
    device_map="cuda",
)

# Apply chat template
tokenizer = get_chat_template(
    tokenizer,
    chat_template="gemma-4",
)

# Set model to evaluation mode
FastModel.for_inference(model)

# ==============================================================================
# 2. Load Unseen / Held-Out Dataset (400-500 samples)
# ==============================================================================
print(f"\n📂 Loading unseen validation samples (indices {EVAL_SAMPLE_START} to {EVAL_SAMPLE_END})...")
eval_dataset = load_dataset(
    "mlabonne/FineTome-100k",
    split=f"train[{EVAL_SAMPLE_START}:{EVAL_SAMPLE_END}]",
)
eval_dataset = standardize_data_formats(eval_dataset)

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [
        tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False).removeprefix("<bos>")
        for convo in convos
    ]
    return {"text": texts}

eval_dataset = eval_dataset.map(formatting_prompts_func, batched=True)
print(f"✅ Prepared {len(eval_dataset)} held-out evaluation samples.")

# ==============================================================================
# 3. Setup SFTTrainer for Standalone Evaluation
# ==============================================================================
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=None,
    eval_dataset=eval_dataset,
    args=SFTConfig(
        dataset_text_field="text",
        per_device_eval_batch_size=2,   # Batch size 2 for fast evaluation
        max_seq_length=MAX_SEQ_LENGTH,
        report_to="none",
        output_dir="eval_temp_output",
    ),
)

# Mask user prompt tokens (calculates loss only on assistant responses)
trainer = train_on_responses_only(trainer)

# ==============================================================================
# 4. Compute Validation Loss & Perplexity
# ==============================================================================
print("\n" + "-" * 70)
print("🚀 Computing validation loss across unseen evaluation samples...")
print("-" * 70)

eval_results = trainer.evaluate()
eval_loss = eval_results.get("eval_loss", 0.0)
perplexity = math.exp(eval_loss) if eval_loss < 20 else float("inf")
gap = eval_loss - FINAL_TRAIN_LOSS

# ==============================================================================
# 5. Overfitting Diagnostic & Generalization Report
# ==============================================================================
print("\n" + "=" * 70)
print("📊 FINAL GENERALIZATION & OVERFITTING REPORT")
print("=" * 70)
print(f"  • Final Training Loss:       {FINAL_TRAIN_LOSS:.4f}")
print(f"  • Held-Out Validation Loss:  {eval_loss:.4f}")
print(f"  • Validation Perplexity:     {perplexity:.2f}")
print(f"  • Generalization Gap (Δ):    {gap:+.4f} (Val Loss - Train Loss)")
print("-" * 70)

if gap <= 0.15:
    status = "🌟 EXCELLENT GENERALIZATION (No Overfitting)"
    analysis = "The validation loss closely matches training loss. The model generalized cleanly."
elif gap <= 0.35:
    status = "✅ HEALTHY / ACCEPTABLE GENERALIZATION"
    analysis = "Normal generalization gap on unseen conversational data. No significant overfitting."
elif gap <= 0.60:
    status = "⚠️ MILD OVERFITTING DETECTED"
    analysis = "Validation loss is noticeably higher than training loss. Consider adding weight decay or reducing epochs."
else:
    status = "🚨 SEVERE OVERFITTING DETECTED"
    analysis = "The model has memorized the training set and struggles on unseen data. Reduce epochs or increase dropout/regularization."

print(f"  Status:   {status}")
print(f"  Analysis: {analysis}")
print("=" * 70 + "\n")
