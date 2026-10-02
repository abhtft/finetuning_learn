import torch
from unsloth import FastModel

# 1. Load base model + trained LoRA adapters
model, tokenizer = FastModel.from_pretrained(
    model_name="gemma_4_e4b_lora",  # Path to saved adapters
    max_seq_length=1024,
    load_in_4bit=True,
    device_map="cuda",
)

# 2. Enable Fast Inference Mode (2x faster generation)
FastModel.for_inference(model)

# 3. Format conversational prompt
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Write essary on indaia"}
        ],
    }
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
).to("cuda")

# 4. Generate Response
outputs = model.generate(
    input_ids=inputs,
    max_new_tokens=256,
    temperature=0.3,
    top_p=0.9,
    use_cache=True,
)

response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
print(response)