# LLM Inference & Deployment Guide

A comprehensive, production-ready guide on how to run inference, evaluate, serve, merge, and deploy fine-tuned Large Language Models (LLMs) trained with **Unsloth**, **LoRA / QLoRA**, and **Hugging Face PEFT**.

This guide covers everything from running ultra-fast local Python generation on your RTX 5050 GPU (8GB VRAM) to serving production APIs with **vLLM**, **Ollama**, **GGUF**, and **Gradio / FastAPI**.

---

## 1. Inference Architecture Overview

When you fine-tune an LLM using LoRA/QLoRA (e.g. via [`train_gemma4_e4b.py`](file:///home/abhishek/projects/unsloth/train_gemma4_e4b.py)), the output folder (such as [`gemma_4_e4b_lora/`](file:///home/abhishek/projects/unsloth/gemma_4_e4b_lora)) contains **only the lightweight adapter matrices** ($\approx 73.5 \text{ MB}$) rather than the entire 16GB base model.

During inference, you have two primary deployment architectures:

```
                  ┌─────────────────────────────────────────────────────────────┐
                  │                 Trained LoRA Adapters (~73.5MB)             │
                  │                   (gemma_4_e4b_lora/)                       │
                  └──────────────────────────────┬──────────────────────────────┘
                                                 │
                   ┌─────────────────────────────┴─────────────────────────────┐
                   │                                                           │
                   ▼                                                           ▼
    ┌──────────────────────────────┐                            ┌──────────────────────────────┐
    │  Path A: Dynamic Adapter     │                            │  Path B: Merged Weights      │
    │  Attachment (On-the-Fly)     │                            │  (Standalone Model)          │
    ├──────────────────────────────┤                            ├──────────────────────────────┤
    │ Base Model (4-bit / 16-bit)  │                            │ Single unified model file:   │
    │               +              │                            │ W_final = W_base + (α/r)(BA) │
    │ LoRA Adapters (r=8)          │                            │                              │
    ├──────────────────────────────┤                            ├──────────────────────────────┤
    │ • Fast local testing         │                            │ • High-throughput serving    │
    │ • Unsloth Native Inference   │                            │ • vLLM / TGI engines         │
    │ • Hugging Face PEFT          │                            │ • GGUF / Ollama / llama.cpp  │
    └──────────────────────────────┘                            └──────────────────────────────┘
```

### VRAM Requirements: Training vs. Inference

Inference requires significantly less VRAM than training because:
1. **No Gradients**: $\nabla W$ memory buffer is 0.
2. **No Optimizer States**: AdamW 8-bit / FP32 tensors are not allocated ($\approx 0 \text{ MB}$).
3. **No Backward Pass Computation Graph**: Activations are discarded immediately after forward attention.

| Model Size | Precision | Training VRAM (QLoRA) | Inference VRAM (KV Cache 1024) | Compatible on 8GB GPU? |
| :--- | :--- | :--- | :--- | :--- |
| **8B (Gemma-4 / Llama-3.1)** | **4-bit (NF4)** | $\approx 7.7 \text{ GB}$ | **$\approx 4.8 \text{ GB}$** |  **Yes (Optimal)** |
| **8B (Gemma-4 / Llama-3.1)** | **8-bit (INT8)** | $> 12 \text{ GB}$ | **$\approx 8.5 \text{ GB}$** | ⚠️ Close to limit |
| **8B (Gemma-4 / Llama-3.1)** | **16-bit (BF16)** | $> 24 \text{ GB}$ | **$\approx 16.0 \text{ GB}$** | ❌ Requires 24GB+ GPU |
| **2B – 3B (Gemma-2 / Qwen-2.5)** | **16-bit (BF16)** | $\approx 7.5 \text{ GB}$ | **$\approx 5.5 \text{ GB}$** |  **Yes** |

---

## 2. Method 1: Native Fast Inference with Unsloth (Recommended for Local Dev)

Unsloth includes custom Triton kernels for the forward pass (`FastModel.for_inference`), boosting generation speed by **up to 2×** compared to standard Hugging Face implementations.

### 2.1 Basic Single-Turn Generation Script

Create a script named `inference_basic.py`:

```python
import torch
from unsloth import FastModel

# 1. Load base model + trained LoRA adapters
model, tokenizer = FastModel.from_pretrained(
    model_name="gemma_4_e4b_lora",  # Path to saved adapter folder
    max_seq_length=1024,
    load_in_4bit=True,
    device_map="cuda",
)

# 2. Enable Unsloth Fast Inference Mode (2x faster decoding)
FastModel.for_inference(model)

# 3. Format conversational prompt using the model's chat template
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Explain the difference between supervised fine-tuning (SFT) and Direct Preference Optimization (DPO)."}
        ]
    }
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,  # Injects <|turn>model\n to prompt the assistant
    return_tensors="pt"
).to("cuda")

# 4. Generate response
with torch.inference_mode():
    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.1,
        use_cache=True,
    )

# 5. Decode only the newly generated response tokens
prompt_length = inputs.shape[1]
response = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)

print("\n--- Assistant Response ---")
print(response)
```

---

### 2.2 Real-Time Streaming Generation (Token-by-Token in Terminal)

For an interactive, snappy feel like ChatGPT in your terminal, use `TextStreamer`:

```python
import torch
from transformers import TextStreamer
from unsloth import FastModel

model, tokenizer = FastModel.from_pretrained(
    model_name="gemma_4_e4b_lora",
    max_seq_length=1024,
    load_in_4bit=True,
    device_map="cuda",
)
FastModel.for_inference(model)

prompt = "Write a concise Python script to scrape top Hacker News headlines using requests and BeautifulSoup."

messages = [
    {
        "role": "user",
        "content": [{"type": "text", "text": prompt}]
    }
]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
).to("cuda")

# Initialize streaming handler
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

print(f"\nUser: {prompt}\n\nAssistant: ", end="", flush=True)

with torch.inference_mode():
    _ = model.generate(
        input_ids=inputs,
        streamer=streamer,
        max_new_tokens=512,
        temperature=0.6,
        top_p=0.9,
        use_cache=True,
    )
print()
```

---

### 2.3 Interactive Multi-Turn Terminal Chatbot

Run an interactive REPL that keeps chat memory in context:

```python
import torch
from transformers import TextStreamer
from unsloth import FastModel

print("Loading fine-tuned model...")
model, tokenizer = FastModel.from_pretrained(
    model_name="gemma_4_e4b_lora",
    max_seq_length=2048,
    load_in_4bit=True,
    device_map="cuda",
)
FastModel.for_inference(model)
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

conversation_history = []

print("\n" + "=" * 50)
print("🤖 Gemma-4 E4B Chatbot Ready! (Type 'exit' or 'clear' to reset)")
print("=" * 50 + "\n")

while True:
    try:
        user_input = input("\nYou: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nExiting...")
        break

    if not user_input:
        continue
    if user_input.lower() in ("exit", "quit"):
        print("Goodbye!")
        break
    if user_input.lower() == "clear":
        conversation_history = []
        print("[Conversation history cleared]")
        continue

    conversation_history.append({
        "role": "user",
        "content": [{"type": "text", "text": user_input}]
    })

    inputs = tokenizer.apply_chat_template(
        conversation_history,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
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

    # Decode and store assistant response in history
    prompt_len = inputs.shape[1]
    assistant_reply = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
    conversation_history.append({
        "role": "assistant",
        "content": [{"type": "text", "text": assistant_reply}]
    })
```

---

## 3. Method 2: Inference with Standard Hugging Face `transformers` + `peft`

If you are running in a production environment where `unsloth` is not installed, you can load the trained LoRA adapters using pure Hugging Face libraries:

```bash
pip install transformers peft bitsandbytes accelerate torch
```

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

adapter_path = "gemma_4_e4b_lora"
base_model_name = "unsloth/gemma-4-E4B-it"  # or google/gemma-4-E4B-it

# 1. 4-bit Quantization Config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# 2. Load Base Model in 4-bit
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map="cuda",
    torch_dtype=torch.bfloat16,
)

# 3. Attach LoRA Adapter
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

# 4. Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(adapter_path)

# 5. Inference
messages = [{"role": "user", "content": "What is Parameter-Efficient Fine-Tuning?"}]
inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(input_ids=inputs, max_new_tokens=256, temperature=0.7)

print(tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True))
```

---

## 4. Method 3: Model Merging for Standalone Deployment

Merging bakes the LoRA weight matrices ($B \times A$) directly into the base weights ($W = W_0 + \frac{\alpha}{r} BA$). This produces a standalone model without any adapter dependencies.

Unsloth provides built-in methods to save in three formats:

```python
from unsloth import FastModel

model, tokenizer = FastModel.from_pretrained(
    model_name="gemma_4_e4b_lora",
    max_seq_length=1024,
    load_in_4bit=True,
    device_map="cuda",
)

# =========================================================================
# Option A: Save Merged 16-bit Model (Best for vLLM, Hugging Face Hub, Cloud)
# =========================================================================
# Requires ~16 GB disk space
model.save_pretrained_merged(
    "gemma_4_e4b_merged_16bit",
    tokenizer,
    save_method="merged_16bit",
)

# =========================================================================
# Option B: Save Merged 4-bit Quantized Model (Compact Standalone)
# =========================================================================
# Requires ~4.5 GB disk space
model.save_pretrained_merged(
    "gemma_4_e4b_merged_4bit",
    tokenizer,
    save_method="merged_4bit",
)

# =========================================================================
# Option C: Push Directly to Hugging Face Hub
# =========================================================================
# model.push_to_hub_merged("your-username/gemma-4-e4b-finetuned", tokenizer, save_method="merged_16bit", token="hf_...")
```

---

## 5. Method 4: High-Throughput Production Serving with vLLM

**vLLM** provides high-performance LLM serving with PagedAttention, continuous batching, and an OpenAI-compatible REST API.

### Step 1: Install vLLM
```bash
pip install vllm
```

### Step 2: Serve the Merged Model or Base Model with LoRA

#### Approach A: Serving Merged 16-bit / 4-bit Model
```bash
vllm serve ./gemma_4_e4b_merged_16bit \
    --port 8000 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 2048 \
    --dtype bfloat16
```

#### Approach B: Serving with Dynamic LoRA Modules
```bash
vllm serve unsloth/gemma-4-E4B-it \
    --enable-lora \
    --lora-modules gemma-custom=./gemma_4_e4b_lora \
    --port 8000 \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.90
```

### Step 3: Querying the OpenAI-Compatible Endpoint

#### Via `curl`:
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-custom",
    "messages": [
      {"role": "user", "content": "How do rocket engines generate thrust in vacuum?"}
    ],
    "temperature": 0.7,
    "max_tokens": 256
  }'
```

#### Via Python `openai` Client:
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-not-required",
)

completion = client.chat.completions.create(
    model="gemma-custom",
    messages=[
        {"role": "system", "content": "You are an expert physics tutor."},
        {"role": "user", "content": "Explain Newton's third law with an example."}
    ],
    temperature=0.7,
    stream=True,
)

for chunk in completion:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()
```

---

## 6. Method 5: Local & Edge Deployment via GGUF (Ollama & llama.cpp)

Exporting your fine-tuned model to **GGUF** enables execution on CPU, Mac (Metal), or lightweight Ollama runners on Windows/Linux.

### Step 1: Export to GGUF using Unsloth

Create a script `export_gguf.py`:

```python
from unsloth import FastModel

model, tokenizer = FastModel.from_pretrained(
    model_name="gemma_4_e4b_lora",
    max_seq_length=1024,
    load_in_4bit=True,
    device_map="cuda",
)

# Export to 4-bit medium quantized GGUF
model.save_pretrained_gguf(
    "gemma_4_e4b_gguf",
    tokenizer,
    quantization_method="q4_k_m",  # Options: q4_k_m, q8_0, f16
)
print("Exported GGUF to ./gemma_4_e4b_gguf/")
```

### Step 2: Create an Ollama `Modelfile`

In the directory where `unsloth.Q4_K_M.gguf` was saved, create a file named `Modelfile`:

```dockerfile
FROM ./gemma_4_e4b_gguf/unsloth.Q4_K_M.gguf

TEMPLATE """<|turn>user
{{ .Prompt }}<end_of_turn>
<|turn>model
{{ .Response }}<end_of_turn>
"""

PARAMETER stop "<end_of_turn>"
PARAMETER stop "<|turn>"
PARAMETER temperature 0.7
PARAMETER top_p 0.9
```

### Step 3: Register and Run with Ollama

```bash
# 1. Build Ollama model
ollama create gemma-finetuned -f Modelfile

# 2. Run interactively
ollama run gemma-finetuned

# 3. Or query via Ollama REST API
curl http://localhost:11434/api/generate -d '{
  "model": "gemma-finetuned",
  "prompt": "Explain gradient descent in 3 bullet points."
}'
```

---

## 7. Method 6: Interactive Web UI (Gradio Chat Interface)

To share a web-based user interface with colleagues or test interactively in the browser:

```bash
pip install gradio
```

Create `app_gradio.py`:

```python
import torch
import gradio as gr
from transformers import TextIteratorer, TextStreamer
from threading import Thread
from unsloth import FastModel

# Load model
print("Loading model for Web UI...")
model, tokenizer = FastModel.from_pretrained(
    model_name="gemma_4_e4b_lora",
    max_seq_length=2048,
    load_in_4bit=True,
    device_map="cuda",
)
FastModel.for_inference(model)

def chat_stream(message, history):
    # Construct conversation history
    messages = []
    for user_msg, bot_msg in history:
        messages.append({"role": "user", "content": [{"type": "text", "text": user_msg}]})
        messages.append({"role": "assistant", "content": [{"type": "text", "text": bot_msg}]})
    messages.append({"role": "user", "content": [{"type": "text", "text": message}]})

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")

    from transformers import TextIteratorStreamer
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generation_kwargs = dict(
        input_ids=inputs,
        streamer=streamer,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    partial_text = ""
    for new_token in streamer:
        partial_text += new_token
        yield partial_text

demo = gr.ChatInterface(
    fn=chat_stream,
    title="Fine-Tuned Gemma-4 E4B Chatbot",
    description="Running locally on NVIDIA RTX 5050 Laptop GPU (8GB VRAM) with Unsloth 4-bit Acceleration.",
    theme="soft",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
```

Run with:
```bash
python app_gradio.py
```
Then open `http://localhost:7860` in your web browser.

---

## 8. Generation Hyperparameter Cheat Sheet

| Parameter | Recommended Range | Purpose & Behavioral Impact |
| :--- | :--- | :--- |
| **`temperature`** | `0.1` – `1.0` | Controls randomness. `0.1`–`0.3` for factual Q&A/code; `0.7`–`0.9` for conversation/creativity; `0.0` for greedy deterministic output. |
| **`top_p` (Nucleus)** | `0.85` – `0.95` | Samples from smallest set of tokens whose cumulative probability $\ge p$. Prevents unlikely tokens while keeping variety. |
| **`top_k`** | `20` – `50` | Filters to the top $K$ most likely next tokens. Filters out tail noise. |
| **`repetition_penalty`** | `1.05` – `1.15` | Penalizes repeating words or phrases already in context. Prevents infinite loops (`1.0` = no penalty). |
| **`max_new_tokens`** | `128` – `1024` | Hard ceiling on output tokens generated. Keep $\le 1024$ to preserve KV-cache memory on 8GB VRAM. |
| **`use_cache`** | `True` | **Crucial**: Caches Key/Value attention activations from previous tokens so complexity is $O(N)$ instead of $O(N^2)$ per generated token. |

### Recommended Presets

```python
# 1. Precise / Factual / Code Generation
deterministic_config = {
    "temperature": 0.2,
    "top_p": 0.95,
    "repetition_penalty": 1.05,
    "max_new_tokens": 512,
}

# 2. Natural Chat & Dialogue
chat_config = {
    "temperature": 0.7,
    "top_p": 0.90,
    "top_k": 40,
    "repetition_penalty": 1.10,
    "max_new_tokens": 512,
}

# 3. Creative / Brainstorming
creative_config = {
    "temperature": 0.9,
    "top_p": 0.95,
    "top_k": 50,
    "repetition_penalty": 1.05,
    "max_new_tokens": 768,
}
```

---

## 9. Deployment Strategy Comparison

| Method | VRAM (8B Model) | Throughput | Setup Complexity | Best For |
| :--- | :--- | :--- | :--- | :--- |
| **Unsloth Native** | **~4.8 GB** | **Fast (35–50 tok/s)** | Very Low | Local development, Python scripting, rapid testing |
| **Hugging Face PEFT** | ~5.2 GB | Moderate (20–30 tok/s) | Low | Generic environments without Unsloth |
| **vLLM (Merged/LoRA)** | ~6.5–8.0 GB | **Ultra Fast (100+ tok/s concurrent)** | Medium | Multi-user REST APIs, production microservices |
| **Ollama / GGUF (Q4_K_M)** | **~4.5 GB** | Fast (30–45 tok/s) | Low | Desktop apps, cross-platform CLI, Mac/CPU edge |
| **Gradio Web UI** | ~4.8 GB | Fast (35–50 tok/s) | Low | Internal company demos, QA testing |

---

## 10. Troubleshooting & Common Inference Issues

### 1. Model repeats the user's prompt or outputs raw tokens like `<|turn>user`
* **Cause**: `tokenizer.decode()` decoded the full sequence including the prompt input tokens.
* **Fix**: Slice the output tensor to decode only the newly generated tokens:
  ```python
  prompt_length = inputs.shape[1]
  response = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
  ```

### 2. Model enters infinite repetition loops
* **Cause**: Missing stop tokens or `repetition_penalty` is too low.
* **Fix**:
  1. Add `repetition_penalty=1.1` to `model.generate()`.
  2. Ensure the tokenizer includes `<end_of_turn>` in `eos_token_id`.

### 3. Generation is extremely slow (< 10 tokens/sec)
* **Checklist**:
  1. Ensure `FastModel.for_inference(model)` was called after loading.
  2. Verify `use_cache=True` is passed to `generate()`.
  3. Wrap the generation call inside `with torch.inference_mode():` to disable autograd tracking.

### 4. CUDA Out of Memory (OOM) on long inputs
* **Fix**:
  1. Keep `max_seq_length` within `1024`–`2048` when loading.
  2. Reduce `max_new_tokens` to `512`.
  3. Ensure `load_in_4bit=True` is enabled.
