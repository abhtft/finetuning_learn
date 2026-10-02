# LLM Deep-Dive: Core Concepts, Mechanics & Training Architecture

A comprehensive technical and experimental reference covering the foundational concepts of Large Language Models (LLMs), fine-tuning mechanics, tokenization, context management, and inference dynamics with **Unsloth**, **LoRA/QLoRA**, and **Hugging Face**.

---

## 1. Context Window & Sliding Window Attention (SWA)

### 1.1 The Classical Attention Problem
In standard multi-head self-attention, every token in a sequence attends to every preceding token. For a sequence of length $L$:
* **Computational Complexity:** $\mathcal{O}(L^2)$
* **Memory (KV Cache) Complexity:** $\mathcal{O}(L \cdot d_{\text{model}} \cdot N_{\text{layers}})$

As context length $L$ grows to 4k, 8k, or 32k tokens, memory and compute requirements scale quadratically during training and linearly during autoregressive generation.

```
Standard Full Attention (Dense N x N Matrix):
Token 1 ──► [ 1 ][ . ][ . ][ . ][ . ][ . ]
Token 2 ──► [ 1 ][ 1 ][ . ][ . ][ . ][ . ]
Token 3 ──► [ 1 ][ 1 ][ 1 ][ . ][ . ][ . ]
Token 4 ──► [ 1 ][ 1 ][ 1 ][ 1 ][ . ][ . ]
Token 5 ──► [ 1 ][ 1 ][ 1 ][ 1 ][ 1 ][ . ]
Token 6 ──► [ 1 ][ 1 ][ 1 ][ 1 ][ 1 ][ 1 ]
```

---

### 1.2 Sliding Window Attention (SWA)
Sliding Window Attention (introduced in models like Mistral, Gemma 2, Gemma 4, and Longformer) restricts each token to attend only to a fixed window of $W$ previous tokens.

* **Local Attention Window:** Token $i$ attends only to tokens from index $\max(0, i - W)$ to $i$.
* **Computational Cost per Layer:** $\mathcal{O}(L \times W)$, reducing quadratic complexity to **linear complexity** relative to sequence length.

```
Sliding Window Attention (Window Size W = 3):
Token 1 ──► [ 1 ][ . ][ . ][ . ][ . ][ . ]
Token 2 ──► [ 1 ][ 1 ][ . ][ . ][ . ][ . ]
Token 3 ──► [ 1 ][ 1 ][ 1 ][ . ][ . ][ . ]
Token 4 ──► [ . ][ 1 ][ 1 ][ 1 ][ . ][ . ]   <-- Token 1 is dropped from attention
Token 5 ──► [ . ][ . ][ 1 ][ 1 ][ 1 ][ . ]   <-- Token 1 & 2 dropped
Token 6 ──► [ . ][ . ][ . ][ 1 ][ 1 ][ 1 ]   <-- Only attends to tokens 4, 5, 6
```

#### Receptive Field Expansion Across Stacked Layers
Even though an individual layer only attends to $W$ tokens, stacked Transformer layers transmit information across wider distances:
* **Layer 1 Receptive Field:** $W$
* **Layer 2 Receptive Field:** $2 \times W$
* **Layer $k$ Receptive Field:** $k \times W$

In a 32-layer model with $W = 4096$, the theoretical receptive field at the final layer is $32 \times 4096 \approx 131\text{k tokens}$.

---

### 1.3 Sliding Window in Multi-Turn Chat Application
In interactive chat loops (e.g. [`infer_chat.py`](file:///home/abhishek/projects/unsloth/infer_chat.py)), context grows continuously with each question and answer.

```
Multi-Turn Sliding Context Window:
Turn 1: [User Q1 + Assistant A1] ───► (Evicted when context exceeds limit)
Turn 2: [User Q2 + Assistant A2]
Turn 3: [User Q3 + Assistant A3]
Turn 4: [User Q4 (Current)] ────────► Preserved in active context
```

To prevent out-of-memory or context-overflow errors:
1. **FIFO Turn Eviction:** Remove the oldest complete user-assistant pairs once token length approaches `max_seq_length`.
2. **System Prompt Pinning:** Keep system instructions (and tool definitions) fixed at index 0, sliding only the conversation history.

---

## 2. Special Characters, Tokens & Tokenization Mechanics

LLMs do not read raw strings; they process numeric **Token IDs** from a fixed vocabulary ($\approx 32\text{k} - 256\text{k}$ tokens).

### 2.1 Anatomy of Special Tokens

| Token Type | Example Notation | Function & Meaning |
| :--- | :--- | :--- |
| **BOS (Beginning of Sequence)** | `<bos>`, `<s>`, `<|begin_of_text|>` | Signals the absolute start of a prompt stream. Injected by the tokenizer. |
| **EOS (End of Sequence)** | `<eos>`, `</s>`, `<|end_of_text|>` | Signals that generation is complete; stops the autoregressive decoding loop. |
| **Turn Start (Chat)** | `<start_of_turn>user\n`, `<|im_start|>user\n` | Delimits speaker boundaries in conversational fine-tuning. |
| **Turn End (Chat)** | `<end_of_turn>`, `<|im_end|>` | Explicitly closes a speaker's response turn. |
| **Generation Prompt** | `<start_of_turn>model\n` | Signals the model to begin generating its completion as the assistant. |
| **PAD (Padding)** | `<pad>`, `[PAD]` | Aligns batches of varying sequence lengths to uniform matrix dimensions. |
| **Reasoning / Thought** | `<thought>`, `</thought>` | Delimits internal Chain-of-Thought reasoning steps before the final answer. |

---

### 2.2 Whitespace, Tab (`\t`), and Multi-Space Tokens
Modern tokenizers (Byte-Pair Encoding / SentencePiece / Unigram) assign dedicated single token IDs to common formatting sequences:
* `\t` $\rightarrow$ Single Tab Token
* `"  "` (2 spaces) $\rightarrow$ Single Token
* `"    "` (4 spaces) $\rightarrow$ Single Token (Common code indentation)
* `"        "` (8 spaces) $\rightarrow$ Single Token

```
String: "def calculate(x):"
Tokens: ["def", " calculate", "(", "x", "):"]

String: "    return x * 2"
Tokens: ["    ", "return", " x", " *", " 2"]  <-- "    " is 1 token!
```

#### The "Gap / Blank In Line" Anomaly Explained
When a model is **undertrained** (e.g., 30 steps) or sampled with **high `repetition_penalty`**:
1. The model predicts a word prefix: `"intellectu"`.
2. Instead of completing `"al"`, the logit for the `\t` (Tab) token or `    ` token becomes slightly higher than other tokens.
3. The model outputs `\t` followed by `"ilization"`.
4. When printed to a terminal emulator via `sys.stdout.write()`, the tab character jumps the cursor to the terminal's **next tab stop column** (e.g., column 80 or 120), creating a wide, visually jarring gap.

---

### 2.3 The "Immediate Blank Response" Anomaly Explained
When an inference script returns `""` (empty string) immediately:
1. The model samples `<eos>` or `<end_of_turn>` as token #1 ($t=0$).
2. The decoder executes:
   ```python
   response = tokenizer.decode([eos_token_id], skip_special_tokens=True) # returns ""
   ```
3. **Mitigation:**
   * Set `min_new_tokens=2` (or `5`) in `model.generate()`.
   * Ensure `add_generation_prompt=True` is passed to `tokenizer.apply_chat_template()`.
   * Ensure previous empty turns are not appended to the history list.

---

## 3. Training Dynamics: Steps, Epochs, Batch Sizes & Schedules

```
                    ┌─────────────────────────────────────────────────────────┐
                    │               Full Dataset (e.g., 500 Samples)          │
                    └────────────────────────────┬────────────────────────────┘
                                                 │
          ┌──────────────────────────────────────┴──────────────────────────────────────┐
          │                                 1 Full Epoch                                │
          ▼                                                                             ▼
   [Batch 1 (Size 4)] ────► [Batch 2 (Size 4)] ────► ... ────► [Batch 125 (Size 4)]
          │
          ▼
   1 Optimizer Step = (per_device_batch_size=1) × (gradient_accumulation_steps=4)
```

### 3.1 Mathematical Definitions

#### 1. Epoch
One complete pass over the entire training dataset.
$$\text{Number of Samples Processed in 1 Epoch} = N_{\text{dataset}}$$

#### 2. Effective Batch Size
The total number of training samples accumulated before updating model weights with an optimizer step:
$$\text{Effective Batch Size} = B_{\text{per\_device}} \times G_{\text{grad\_accum}} \times N_{\text{GPUs}}$$

#### 3. Total Training Steps
The total number of weight update cycles executed:
$$\text{Total Steps} = \left\lceil \frac{N_{\text{dataset}}}{\text{Effective Batch Size}} \right\rceil \times \text{Epochs}$$

---

### 3.2 Comparison: Steps vs. Epochs Configuration

| Configuration | What it does | When to use |
| :--- | :--- | :--- |
| `max_steps = 30` | Stops training exactly at 30 optimizer steps, regardless of dataset size. | **Smoke testing**, checking VRAM allocation, verifying loss decrease, benchmarking TFLOPS. |
| `num_train_epochs = 1` | Trains until every sample in the dataset has been seen exactly once. | Standard production fine-tuning on clean, balanced instruction datasets. |
| `num_train_epochs = 3` | Passes over the dataset 3 times. | Small datasets ($<1,000$ samples) or classification/extraction tasks requiring high memorization. |

> [!WARNING]
> **The 30-Step Smoke Test Trap:** 30 steps on 500 samples (with effective batch size 4) only sees $30 \times 4 = 120$ samples ($24\%$ of 1 epoch). The model has not converged and will suffer from subword stuttering, hallucinations, and formatting glitches.

---

### 3.3 Learning Rate Schedulers & Warmup

```
Learning
Rate (η)
  ▲
  │        /\  (Peak LR: e.g. 2e-4)
  │       /  \
  │      /    \
  │     /      \─── (Cosine / Linear Decay)
  │    /
  │   / (Warmup Steps: 5 - 10% of total)
  └───┴────────────────────────────────────────► Training Steps
```

* **Warmup Steps (`warmup_steps = 5` to `50`):** Linearly ramps up learning rate from 0 to peak LR ($2\times 10^{-4}$) to prevent large initial gradients from destroying pre-trained base model representations.
* **Decay Schedule (`lr_scheduler_type = "cosine"` / `"linear"`):** Gradually decays the learning rate to 0 as training nears completion, allowing weights to settle into optimal local minima.

---

## 4. Response Masking (`train_on_responses_only`)

Standard pre-training trains on every token in the sequence (next-token prediction on both prompt and response). 

In **Instruction Fine-Tuning (SFT)**, computing loss on the user's prompt is counterproductive because:
1. The model should learn how to **answer**, not how to ask user questions.
2. Training on user prompts wastes gradient capacity on input phrasing variance.

```
Full Sequence:
<start_of_turn>user\nWrite an essay on India.<end_of_turn>\n<start_of_turn>model\nIndia is a vibrant...<end_of_turn>

Without Masking (Standard Causal LM):
Loss Computed: [  ██████████████████████████████████████████████████████████████████████████████████████████████████████████████ ]

With Response Masking (train_on_responses_only):
Labels Array:  [ -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, "India", " is", " a", " vibrant", ... , <end_of_turn> ]
Loss Computed: [ ----------------- MASKED (Ignored by PyTorch) ----------------- ][ ██████████ GRADIENTS COMPUTED ██████████ ]
```

* In PyTorch CrossEntropyLoss, the index `-100` is the default `ignore_index`.
* [`train_on_responses_only(trainer)`](file:///home/abhishek/projects/unsloth/train_gemma4_e4b.py#L81) in Unsloth scans the token stream and assigns `-100` to all tokens prior to `<start_of_turn>model\n`.

---

## 5. Decoding Dynamics & Inference Hyperparameters

During text generation, logits (unnormalized log-probabilities $\mathbf{z}$) from the final Transformer layer are converted into probabilities via Softmax:

$$P(w_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

```
Vocabulary Logits ──► [ Temperature Scaling (z / T) ] ──► [ Repetition Penalty ] ──► [ Top-K / Top-P Filtering ] ──► [ Categorical Sampling ] ──► Next Token ID
```

### 5.1 Hyperparameter Matrix

| Parameter | Recommended Range | Mechanism | Impact of Setting Too High | Impact of Setting Too Low |
| :--- | :--- | :--- | :--- | :--- |
| **`temperature`** | `0.1 – 0.7` | Scales the logit distribution before softmax. | Rambling, hallucinations, incoherence ($T > 1.0$). | Deterministic, repetitive, robotic ($T = 0$). |
| **`top_p` (Nucleus)** | `0.8 – 0.95` | Accumulates top tokens until cumulative probability exceeds $p$. | Includes low-probability outlier tokens. | Limits vocabulary variety; overly rigid phrasing. |
| **`top_k`** | `20 – 50` | Retains only the $K$ highest-probability tokens. | Allows noise from vocabulary tail. | Cuts off plausible creative alternatives. |
| **`repetition_penalty`** | `1.0 – 1.05` | Divides logits of previously generated tokens by $\theta > 1.0$. | Forces model to pick bizarre synonyms, spaces, or tabs. | Model gets stuck in infinite repeating loops. |
| **`min_new_tokens`** | `1 – 5` | Forbids the EOS token until minimum token count is met. | May produce filler text for single-word queries. | Can produce instant empty/blank responses. |
| **`max_new_tokens`** | `256 – 2048` | Hard ceiling on output token length. | Risk of VRAM exhaustion if KV cache exceeds memory. | Truncates sentences mid-thought. |

---

## 6. LoRA & QLoRA Mathematical Foundation

Instead of updating all billions of parameters in weight matrix $W_0 \in \mathbb{R}^{d \times k}$, **LoRA** decomposes the update $\Delta W$ into two low-rank matrices $A$ and $B$:

$$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} (B \cdot A)$$

Where:
* $B \in \mathbb{R}^{d \times r}$ initialized to $0$
* $A \in \mathbb{R}^{r \times k}$ initialized with Gaussian random values ($\mathcal{N}(0, \sigma^2)$)
* $r \ll \min(d, k)$ (Rank, e.g., $r = 8$)
* $\alpha$ is the constant scaling factor (e.g., $\alpha = 8$ or $16$)

```
                     x (Input Vector)
                    ┌┴┐
                    │ │
        ┌───────────┴─┴───────────┐
        │                         │
        ▼                         ▼
┌───────────────┐         ┌───────────────┐
│               │         │  Matrix A     │ (Rank r = 8)
│  Frozen Base  │         └───────┬───────┘
│  Weights (W0) │                 │
│  (4-bit NF4)  │                 ▼
│               │         ┌───────────────┐
│               │         │  Matrix B     │ (Rank r = 8)
└───────┬───────┘         └───────┬───────┘
        │                         │
        │                         ▼
        │                  Scaled by (α/r)
        │                         │
        └───────────┬─────────────┘
                    ▼
                 y = W0·x + (α/r)·B·A·x
```

### 6.1 Why QLoRA Fits on 8GB GPUs
1. **Base Weights $W_0$ in 4-bit NF4 (NormalFloat4):** Uses an information-theoretically optimal distribution for normally distributed weights, shrinking an 8B model from 16GB down to **4.8GB VRAM**.
2. **Double Quantization (DQ):** Quantizes the quantization constants themselves, saving $\approx 0.37 \text{ bits/param}$ ($\approx 370\text{MB}$ on 8B models).
3. **Paged Optimizers:** Offloads momentary optimizer memory spikes to CPU RAM during extreme gradient bursts.

---

## 7. Diagnostic Cheatsheet: Symptoms, Root Causes & Fixes

```
┌──────────────────────────────────────┬──────────────────────────────────────────┬──────────────────────────────────────────────────────┐
│ Symptom Observed                     │ Root Cause                               │ Actionable Fix                                       │
├──────────────────────────────────────┼──────────────────────────────────────────┼──────────────────────────────────────────────────────┤
│ Response is completely blank ("")    │ Model output EOS token at step t=0       │ Set min_new_tokens=2; check add_generation_prompt=True│
├──────────────────────────────────────┼──────────────────────────────────────────┼──────────────────────────────────────────────────────┤
│ Huge horizontal tab gaps in lines    │ \t / space tokens sampled + rep_penalty  │ Set rep_penalty=1.02; lower temperature; train >100s │
├──────────────────────────────────────┼──────────────────────────────────────────┼──────────────────────────────────────────────────────┤
│ CUDA Out of Memory (OOM) on step 1   │ max_seq_length too high / batch too big  │ Lower max_seq_length to 1024; set batch_size=1       │
├──────────────────────────────────────┼──────────────────────────────────────────┼──────────────────────────────────────────────────────┤
│ Loss remains flat (does not drop)    │ Learning rate too low or all text masked │ Verify lr >= 1e-4; check train_on_responses_only tags│
├──────────────────────────────────────┼──────────────────────────────────────────┼──────────────────────────────────────────────────────┤
│ Model repeats words endlessly        │ Repetition penalty 1.0 + low temperature │ Set repetition_penalty=1.05; temperature=0.5         │
├──────────────────────────────────────┼──────────────────────────────────────────┼──────────────────────────────────────────────────────┤
│ Conversation history ignored         │ Chat template role syntax incorrect      │ Use tokenizer.apply_chat_template with proper schema │
└──────────────────────────────────────┴──────────────────────────────────────────┴──────────────────────────────────────────────────────┘
```

---

## 8. Summary of Best Practices for Production

1. **Token Budgeting:** Always calculate $L_{\text{prompt}} + L_{\text{max\_new\_tokens}} \le \text{max\_seq\_length}$ to prevent silent attention truncation.
2. **Sanitize Streamers:** In interactive CLI apps, sanitize control characters (`\t`, `\r`) in custom streamer callbacks.
3. **Training Duration:** Never evaluate a model trained for only 30 steps; complete at least **1 full epoch** or $200+$ optimizer steps for stable token representations.
4. **Adapter Separation:** Keep lightweight LoRA adapters ($\approx 73\text{MB}$) distinct from the base model for rapid iteration, merging only for high-throughput deployment servers.
