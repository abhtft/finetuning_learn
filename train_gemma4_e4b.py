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
print("Loading dataset...")
dataset = load_dataset("mlabonne/FineTome-100k", split="train[:500]")  # 500 rows for quick benchmark run
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
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=30,                     # Set to 30 steps for initial speed/memory test
        learning_rate=2e-4,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        seed=3407,
        report_to="none",
        output_dir="gemma_4_e4b_output",
    ),
)

# 6. Mask user prompt to train only on assistant responses
trainer = train_on_responses_only(trainer)

# 7. Start Training
print("Starting fine-tuning...")
trainer_stats = trainer.train()

# 8. Save LoRA Adapters
print("Saving LoRA adapters...")
model.save_pretrained("gemma_4_e4b_lora")
tokenizer.save_pretrained("gemma_4_e4b_lora")
print("Training finished and adapters saved.")