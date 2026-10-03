# Production Deployment Guide: Gemma-4 E4B LoRA Adapter

A step-by-step practitioner guide for deploying your fine-tuned **`gemma_4_e4b_lora`** model across the machine learning ecosystem — from **Hugging Face Hub** and **Ollama / GGUF** to high-throughput production engines like **vLLM** and **Hugging Face Inference Endpoints**.

---

## 📑 Table of Contents
1. [Prerequisites & Hugging Face Authentication](#1-prerequisites--hugging-face-authentication)
2. [Method 1: Upload Lightweight LoRA Adapters to Hugging Face Hub](#method-1-upload-lightweight-lora-adapters-to-hugging-face-hub)
3. [Method 2: Export & Deploy GGUF to Ollama & llama.cpp (Local / Edge)](#method-2-export--deploy-gguf-to-ollama--llamacpp-local--edge)
4. [Method 3: Merge into Full 16-Bit Model & Serve with vLLM](#method-3-merge-into-full-16-bit-model--serve-with-vllm)
5. [Method 4: Deploy to Cloud Managed Endpoints (HF Endpoints / RunPod / Modal)](#method-4-deploy-to-cloud-managed-endpoints-hf-endpoints--runpod--modal)
6. [Method 5: Interactive Web UI on Hugging Face Spaces (Gradio)](#method-5-interactive-web-ui-on-hugging-face-spaces-gradio)
7. [Deployment Strategy Matrix (Which Option to Choose?)](#7-deployment-strategy-matrix)

---

## 1. Prerequisites & Hugging Face Authentication

To upload models or adapters to Hugging Face, generate an **Access Token** with **Write permissions** from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

Authenticate in your terminal:
```bash
# In your active environment (gemma_env)
huggingface-cli login
```
*(Paste your Hugging Face write token when prompted).*

---

## Method 1: Upload Lightweight LoRA Adapters to Hugging Face Hub

* **Best For:** Quick sharing, minimal bandwidth (~50MB upload), and modular adapter switching.
* **Storage Size:** ~50 – 100 MB.

### Python Script: `upload_lora_to_hf.py`

Create and run the following script:

```python
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
```

### How End-Users Load Your Adapter:
```python
from unsloth import FastModel

model, tokenizer = FastModel.from_pretrained(
    model_name="your_hf_username/gemma-4-E4B-it-finetome-lora",
    max_seq_length=1024,
    load_in_4bit=True,
)
FastModel.for_inference(model)
```

---

## Method 2: Export & Deploy GGUF to Ollama & llama.cpp (Local / Edge)

* **Best For:** Offline local execution on Mac, Windows, Linux, and low-spec consumer laptops without cloud dependencies.
* **Formats Supported:** `q4_k_m` (recommended 4-bit), `q5_k_m`, `q8_0`, `f16`.

### Step 2.1: Export to GGUF using Unsloth

Create and run `export_to_gguf.py`:

```python
from unsloth import FastModel

model, tokenizer = FastModel.from_pretrained(
    model_name="gemma_4_e4b_lora",
    max_seq_length=1024,
    dtype=None,
    load_in_4bit=True,
)

# Export directly to 4-bit GGUF
print("📦 Exporting merged model to GGUF format (Q4_K_M)...")
model.save_pretrained_gguf(
    "gemma_4_e4b_gguf",
    tokenizer,
    quantization_method="q4_k_m",  # Optimal balance of speed and quality
)

# Optional: Push GGUF directly to Hugging Face Hub
# model.push_to_hub_gguf("your_hf_username/gemma-4-E4B-GGUF", tokenizer, quantization_method="q4_k_m")
print("✅ GGUF exported to directory: gemma_4_e4b_gguf/")
```

### Step 2.2: Run with Ollama

1. Create a `Modelfile` in the directory containing your `.gguf` file:
   ```dockerfile
   FROM ./gemma_4_e4b_gguf/unsloth.Q4_K_M.gguf

   TEMPLATE """{{ if .System }}<|turn>system
   {{ .System }}<|end_of_turn>
   {{ end }}{{ range .Messages }}<|turn>{{ .Role }}
   {{ .Content }}<|end_of_turn>
   {{ end }}<|turn>model
   """

   PARAMETER temperature 0.7
   PARAMETER top_p 0.9
   PARAMETER stop "<|end_of_turn>"
   ```

2. Register and start with Ollama:
   ```bash
   # Build the Ollama model
   ollama create gemma4-custom -f Modelfile

   # Chat with it interactively
   ollama run gemma4-custom
   ```

---

## Method 3: Merge into Full 16-Bit Model & Serve with vLLM

* **Best For:** High-throughput production APIs (serving 100+ concurrent requests/sec with PagedAttention and continuous batching).
* **Storage Size:** ~16 GB (full FP16/BF16 model).

### Step 3.1: Export Merged 16-Bit Weights

Create and run `merge_16bit.py`:

```python
from unsloth import FastModel

model, tokenizer = FastModel.from_pretrained(
    model_name="gemma_4_e4b_lora",
    max_seq_length=1024,
    dtype=None,
    load_in_4bit=True,
)

# Merge LoRA weights back into the base model and save as 16-bit
print("🔄 Merging LoRA into base weights (16-bit)...")
model.save_pretrained_merged(
    "gemma_4_e4b_merged_16bit",
    tokenizer,
    save_method="merged_16bit",
)

# Push merged model to Hugging Face Hub
HF_USERNAME = "your_hf_username"
model.push_to_hub_merged(
    f"{HF_USERNAME}/gemma-4-E4B-it-finetome-merged",
    tokenizer,
    save_method="merged_16bit",
    private=False,
)
print("✅ Merged 16-bit model saved and pushed to Hugging Face!")
```

### Step 3.2: Serve via vLLM (OpenAI-Compatible REST API)

Install and launch the vLLM server:
```bash
pip install vllm

# Start OpenAI-compatible API server on port 8000
python -m vllm.entrypoints.openai.api_server \
    --model gemma_4_e4b_merged_16bit \
    --gpu-memory-utilization 0.90 \
    --max-model-len 1024 \
    --port 8000
```

### Step 3.3: Call your API via standard OpenAI client:
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")

response = client.chat.completions.create(
    model="gemma_4_e4b_merged_16bit",
    messages=[
        {"role": "user", "content": "Explain the concept of entropy in simple terms."}
    ],
    temperature=0.7,
)
print(response.choices[0].message.content)
```

---

## Method 4: Deploy to Cloud Managed Endpoints (HF Endpoints / RunPod / Modal)

### Option A: Hugging Face Inference Endpoints (1-Click Serverless)
1. Push your merged model or adapter to Hugging Face Hub (as done in Method 1 or 3).
2. Go to [ui.endpoints.huggingface.co](https://ui.endpoints.huggingface.co).
3. Click **"New Endpoint"** and select your repository (`your_hf_username/gemma-4-E4B-it-finetome-merged`).
4. Select GPU Instance: **NVIDIA A10G (24GB)** or **L4 (24GB)**.
5. Under Framework, select **Text Generation Inference (TGI)** or **vLLM**.
6. Click **Create Endpoint** $\rightarrow$ You receive a production HTTPS endpoint URL with auto-scaling and zero DevOps management.

### Option B: Serverless Deployment on Modal
```python
# modal_deploy.py
import modal

app = modal.App("gemma-4-e4b-service")
image = modal.Image.debian_slim().pip_install("vllm", "torch")

@app.function(image=image, gpu="A10G", container_idle_timeout=300)
@modal.web_server(port=8000)
def serve():
    import subprocess
    subprocess.Popen([
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", "your_hf_username/gemma-4-E4B-it-finetome-merged",
        "--port", "8000",
    ])
```

Deploy with: `modal deploy modal_deploy.py`

---

## Method 5: Interactive Web UI on Hugging Face Spaces (Gradio)

Create a free, shareable interactive web demo hosted directly on Hugging Face Spaces.

### `app.py` for Hugging Face Spaces:
```python
import gradio as gr
import torch
from unsloth import FastModel

# Load model
model, tokenizer = FastModel.from_pretrained(
    model_name="your_hf_username/gemma-4-E4B-it-finetome-lora",
    max_seq_length=1024,
    load_in_4bit=True,
)
FastModel.for_inference(model)

def chat(message, history):
    messages = []
    for user_msg, bot_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "model", "content": bot_msg})
    messages.append({"role": "user", "content": message})
    
    inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.05,
        )
    response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    return response

demo = gr.ChatInterface(fn=chat, title="🤖 Fine-Tuned Gemma-4 E4B Chatbot")
demo.launch()
```

---

## 7. Deployment Strategy Matrix

| Method | Latency | Throughput | Ease of Setup | Cost | Best Use Case |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **HF Hub LoRA Upload** | N/A | N/A | ⭐⭐⭐⭐⭐ (Easiest) | Free | Open-source sharing & versioning |
| **Ollama / GGUF** | Low | Low–Medium | ⭐⭐⭐⭐⭐ (Very Easy) | Free (Local) | Desktop apps, private offline use |
| **vLLM (Merged FP16)** | Ultra-Low | Ultra-High | ⭐⭐⭐☆ (Moderate) | Server cost | Production SaaS, concurrent APIs |
| **HF Inference Endpoints**| Ultra-Low | High | ⭐⭐⭐⭐⭐ (1-Click) | Pay-per-sec | Managed production enterprise APIs |
| **HF Spaces (Gradio)** | Medium | Low | ⭐⭐⭐⭐ (Easy) | Free / Low | Public portfolio demos & QA testing |
