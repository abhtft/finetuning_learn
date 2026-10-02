import sys
import argparse
import torch
from transformers import TextStreamer
from unsloth import FastModel

def main():
    parser = argparse.ArgumentParser(description="Interactive Terminal Chatbot for Fine-Tuned Gemma-4")
    parser.add_argument("--model-path", type=str, default="gemma_4_e4b_lora", help="Path to LoRA adapter folder or base model")
    parser.add_argument("--max-seq-length", type=int, default=2048, help="Maximum sequence context length (default: 2048)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (default: 0.7)")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus top_p sampling (default: 0.9)")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max new tokens to generate per response (default: 512)")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("🚀 Initializing Gemma-4 Chatbot...")
    print(f"📁 Loading model from: '{args.model_path}'")
    print("=" * 60 + "\n")

    # 1. Load model and tokenizer/processor in 4-bit mode
    model, tokenizer = FastModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        device_map="cuda",
    )

    # 2. Enable Unsloth Fast Inference Mode (2x faster decoding)
    FastModel.for_inference(model)

    # 3. Setup streaming output for real-time terminal responses
    # Use the inner text tokenizer if wrapped in a multimodal processor
    text_tokenizer = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
    streamer = TextStreamer(text_tokenizer, skip_prompt=True, skip_special_tokens=True)

    conversation_history = []

    print("\n" + "=" * 60)
    print("🤖 Gemma-4 Chatbot Ready!")
    print("Commands:")
    print("  • 'exit' or 'quit'  : Exit the chat session")
    print("  • 'clear' or 'reset': Clear conversation memory")
    print("  • 'history'         : Display conversation history summary")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("\033[1;36mYou:\033[0m ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye! 👋")
            break

        if not user_input:
            continue

        cmd = user_input.lower()
        if cmd in ("exit", "quit", "q"):
            print("\nGoodbye! 👋")
            break

        if cmd in ("clear", "reset"):
            conversation_history = []
            print("\n\033[1;33m[Conversation history cleared]\033[0m\n")
            continue

        if cmd == "history":
            turns = len(conversation_history) // 2
            print(f"\n\033[1;33m[Current history: {turns} conversation turns ({len(conversation_history)} messages)]\033[0m\n")
            continue

        # Append user turn (multimodal typed block format for Gemma4Processor)
        conversation_history.append({
            "role": "user",
            "content": [{"type": "text", "text": user_input}]
        })

        # Apply chat template
        try:
            inputs = tokenizer.apply_chat_template(
                conversation_history,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to("cuda")
        except Exception as e:
            print(f"\n\033[1;31mError applying chat template:\033[0m {e}")
            conversation_history.pop()
            continue

        print("\n\033[1;32mAssistant:\033[0m ", end="", flush=True)

        try:
            with torch.inference_mode():
                outputs = model.generate(
                    input_ids=inputs,
                    streamer=streamer,
                    max_new_tokens=args.max_tokens,
                    temperature=0.3,
                    top_p=args.top_p,
                    repetition_penalty=1.05,
                    use_cache=True,
                )

            # Extract generated response tokens to append to conversation history
            prompt_len = inputs.shape[1]
            assistant_reply = text_tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True).strip()
            
            conversation_history.append({
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_reply}]
            })
            print()

        except KeyboardInterrupt:
            print("\n\033[1;33m[Generation interrupted by user]\033[0m\n")
            conversation_history.pop()
        except Exception as e:
            print(f"\n\033[1;31mGeneration error:\033[0m {e}\n")
            conversation_history.pop()

if __name__ == "__main__":
    main()
