from unsloth import FastModel

# 1. Load the fine-tuned adapter and base model
model, tokenizer = FastModel.from_pretrained(
    model_name="gemma_4_e4b_lora",
    max_seq_length=1024,
    dtype=None,
    load_in_4bit=True,
)

# 2. Push adapter weights and tokenizer to Hugging Face Hub
HF_USERNAME = "your_hf_username"          # Replace with your Hugging Face username
REPO_NAME   = "gemma-4-E4B-it-finetome-lora"

print(f"🚀 Pushing LoRA adapters to {HF_USERNAME}/{REPO_NAME}...")
model.push_to_hub(
    f"{HF_USERNAME}/{REPO_NAME}",
    token=True,
    private=False,  # Set to True for private repo
)
tokenizer.push_to_hub(
    f"{HF_USERNAME}/{REPO_NAME}",
    token=True,
    private=False,
)
print("✅ LoRA adapters uploaded successfully!")