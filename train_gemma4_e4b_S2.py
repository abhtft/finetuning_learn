import torch
from unsloth import FastModel
from datasets import load_dataset
from unsloth.chat_templates import get_chat_template, standardize_data_formats, train_on_responses_only
from trl import SFTTrainer, SFTConfig

# 1. Hardware & Model Configuration
max_seq_length = 1024  # Fits comfortably within 8GB VRAM
load_in_4bit = True


#
model, tokenizer = FastModel.from_pretrained(
    model_name="unsloth/gemma-4-E4B-it",
    dtype=None,  # Auto-detects bfloat16 for RTX 50-series
    max_seq_length=max_seq_length,
    load_in_4bit=load_in_4bit,
    full_finetuning=False,
    device_map="cuda",
)

# 2. Add LoRA Adapters
model = FastModel.get_peft_model(
    model,
    finetune_vision_layers=False,       # Turn off to save VRAM for text fine-tuning
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=8,
    lora_alpha=8,
    lora_dropout=0,
    bias="none",
    random_state=3407,
)

# 3. Apply Gemma-4 Chat Template
tokenizer = get_chat_template(
    tokenizer,
    chat_template="gemma-4",
)

# 4. Load & Prepare Dataset (FineTome-100k)
# 1,200 samples is optimal for student learning: ~150 steps, ~25 mins runtime, full loss curve
SAMPLE_SIZE = 1200
print(f"Loading {SAMPLE_SIZE} samples from dataset for educational fine-tuning...")
dataset = load_dataset("mlabonne/FineTome-100k", split=f"train[:{SAMPLE_SIZE}]")
dataset = standardize_data_formats(dataset)

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [
        tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False).removeprefix("<bos>")
        for convo in convos
    ]
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True)

# 5. Trainer Configuration (Optimized for 8GB Mobile GPU)
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    eval_dataset=None,
    args=SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,  # Effective Batch Size = 8
        num_train_epochs=1,             # 1 full epoch (1,200 / 8 = 150 total steps)
        warmup_ratio=0.06,              # 6% warmup (~9 steps)
        learning_rate=1.8e-4,
        logging_steps=5,                # Log loss every 5 steps for smooth curve visualization
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",     # Cosine decay over the full run
        seed=3407,
        report_to="none",
        output_dir="gemma_4_e4b_output",
    ),
)

# 6. Mask user prompt to train only on assistant responses
trainer = train_on_responses_only(trainer)

# 7. Start Training
print("\n" + "=" * 70)
print(f"🔥 Starting Training: 1 Full Epoch on {len(dataset)} Samples (~{len(dataset)//8} Steps)")
print("=" * 70 + "\n")
trainer_stats = trainer.train()

# 8. Educational Loss Variation & Training Analytics Summary
print("\n" + "=" * 70)
print("📊 EDUCATIONAL LOSS VARIATION & CONVERGENCE REPORT")
print("=" * 70)

log_history = [log for log in trainer.state.log_history if "loss" in log]

if log_history:
    initial_loss = log_history[0]["loss"]
    final_loss = log_history[-1]["loss"]
    min_loss = min(log["loss"] for log in log_history)
    loss_drop = ((initial_loss - final_loss) / initial_loss) * 100

    print(f"\n📈 Summary Metrics:")
    print(f"  • Total Dataset Samples:   {len(dataset)}")
    print(f"  • Effective Batch Size:    8 (1 per-device × 8 grad accum)")
    print(f"  • Total Completed Steps:   {trainer.state.global_step}")
    print(f"  • Initial Loss (Step {log_history[0]['step']}):   {initial_loss:.4f}")
    print(f"  • Final Loss (Step {log_history[-1]['step']}):     {final_loss:.4f}")
    print(f"  • Lowest Loss Observed:    {min_loss:.4f}")
    print(f"  • Total Loss Reduction:    {loss_drop:.1f}%\n")

    print("📋 Step-by-Step Loss Progression:")
    print(f"  {'Step':<8} | {'Epoch':<8} | {'Loss':<10} | {'LR':<12} | {'Visual Trend'}")
    print("  " + "-" * 62)

    max_l = max(log["loss"] for log in log_history)
    min_l = min(log["loss"] for log in log_history)

    for log in log_history:
        step = log.get("step", 0)
        epoch = log.get("epoch", 0.0)
        loss = log.get("loss", 0.0)
        lr = log.get("learning_rate", 0.0)
        
        # ASCII visualization bar (shorter bar = lower loss = better learning)
        normalized = int(20 * (loss - min_l) / (max_l - min_l + 1e-8)) if max_l != min_l else 10
        bar = "█" * (normalized + 1)
        
        print(f"  {step:<8} | {epoch:<8.2f} | {loss:<10.4f} | {lr:<12.2e} | {bar}")

print("\n" + "=" * 70)

# 9. Save LoRA Adapters
print("💾 Saving fine-tuned LoRA adapters to 'gemma_4_e4b_lora'...")
model.save_pretrained("gemma_4_e4b_lora")
tokenizer.save_pretrained("gemma_4_e4b_lora")
print("✅ Training finished and adapters saved successfully!")