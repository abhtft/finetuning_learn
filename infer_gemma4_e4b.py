import torch
from transformers import TextStreamer
from unsloth import FastModel

# 1. Hardware & Model Configuration
model_path = "gemma_4_e4b_lora"
max_seq_length = 1024

print("=" * 60)
print(f"Loading fine-tuned model and tokenizer from '{model_path}'...")
print("=" * 60)

model, tokenizer = FastModel.from_pretrained(
    model_name=model_path,
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    device_map="cuda",
)

# 2. Enable Fast Inference Mode (2x faster decoding via Triton kernels)
FastModel.for_inference(model)

# 3. Setup Token Streamer for real-time output
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

# 4. Interactive Chat Loop
conversation_history = []

print("\n" + "=" * 60)
print(" Gemma-4 E4B Fine-Tuned Chatbot Ready!")
print(" Type your prompt and press Enter.")
print(" Commands: 'exit' to quit | 'clear' to reset chat history.")
print("=" * 60 + "\n")

while True:
    try:
        user_input = input("You: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nSession ended.")
        break

    if not user_input:
        continue
    if user_input.lower() in ("exit", "quit", "q"):
        print("Exiting...")
        break
    if user_input.lower() in ("clear", "reset"):
        conversation_history = []
        print("\n[Chat history cleared]\n")
        continue

    # Append user turn
    conversation_history.append({"role": "user", "content": user_input})

    # Apply Gemma-4 chat template
    inputs = tokenizer.apply_chat_template(
        conversation_history,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    print("\nAssistant: ", end="", flush=True)
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=inputs,
            streamer=streamer,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            use_cache=True,
        )

    # Extract generated tokens and record in history
    prompt_len = inputs.shape[1]
    assistant_reply = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
    conversation_history.append({"role": "assistant", "content": assistant_reply})
    print()
