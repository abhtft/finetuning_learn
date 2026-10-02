# LLM Fine-Tuning: Iterative Engineering & Acceptance Criteria Framework

A comprehensive, production-grade guide on why **iterative development** is essential for Large Language Model (LLM) fine-tuning and how to define, measure, and enforce **Acceptance Criteria** before deploying fine-tuned models to production.

---

## 1. Why Iterative Development is Essential in LLM Engineering

In traditional software engineering, code either compiles or throws an error. In LLM fine-tuning, training can complete with zero errors while producing a model that is completely unusable due to hallucinations, formatting degeneration, tone drift, or catastrophic forgetting.

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 The Iterative Fine-Tuning Flywheel                                     │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘

              ┌─────────────────────────────────────────────────────────────┐
              │ 1. Data Curation & Strict Sanitization (100 - 500 Samples)  │
              └──────────────────────────────┬──────────────────────────────┘
                                             │
                                             ▼
              ┌─────────────────────────────────────────────────────────────┐
              │ 2. Rapid Prototype Run (Smoke Test: 30 - 100 Steps)         │
              └──────────────────────────────┬──────────────────────────────┘
                                             │
                                             ▼
              ┌─────────────────────────────────────────────────────────────┐
              │ 3. Qualitative Failure Mode Analysis (Manual Edge Cases)    │
              └──────────────────────────────┬──────────────────────────────┘
                                             │
                                             ▼
              ┌─────────────────────────────────────────────────────────────┐
              │ 4. Data Refinement & Hard-Negative Injection                │
              └──────────────────────────────┬──────────────────────────────┘
                                             │
                                             ▼
              ┌─────────────────────────────────────────────────────────────┐
              │ 5. Scaled Training & Hyperparameter Sweep (1 - 3 Epochs)    │
              └──────────────────────────────┬──────────────────────────────┘
                                             │
                                             ▼
              ┌─────────────────────────────────────────────────────────────┐
              │ 6. Automated Golden Eval & LLM-as-a-Judge Benchmarking      │
              └──────────────────────────────┬──────────────────────────────┘
                                             │
                                             ▼
              ┌─────────────────────────────────────────────────────────────┐
              │ 7. Go / No-Go Deployment Decision Gate                      │
              └─────────────────────────────────────────────────────────────┘
```

### The "Big Bang" Fallacy vs. Rapid Iteration

| Approach | Typical Workflow | Failure Risk | Resource Waste |
| :--- | :--- | :--- | :--- |
| **Big Bang (One-Shot)** | Collect 50,000 unverified samples $\rightarrow$ Train 50 hours on cloud GPU $\rightarrow$ Discover format bugs on day 3. | **Extremely High** (Subtle dataset errors corrupt entire run). | Huge compute spend, wasted days. |
| **Iterative (Agile)** | Clean 400 samples $\rightarrow$ Train 20 mins $\rightarrow$ Inspect failures $\rightarrow$ Fix template $\rightarrow$ Scale to 5k $\rightarrow$ Benchmark. | **Minimal** (Failures identified within 15–30 mins). | Low compute, rapid learning. |

---

## 2. Comprehensive Acceptance Criteria Framework

Before any fine-tuned model is deemed "production-ready," it must satisfy five discrete acceptance gates:

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                     The 5 Acceptance Gates Matrix                                      │
├───────────────────┬───────────────────┬───────────────────┬───────────────────┬────────────────────────┤
│ Gate 1: Technical │ Gate 2: Task      │ Gate 3: Structural│ Gate 4: Latency & │ Gate 5: Safety &       │
│ Convergence       │ Accuracy          │ Compliance        │ Performance       │ Alignment              │
├───────────────────┼───────────────────┼───────────────────┼───────────────────┼────────────────────────┤
│ • Eval Loss < 0.60│ • Win Rate > 70%  │ • Schema Valid    │ • TTFT < 300ms    │ • Hallucination < 3%   │
│ • No Overfitting  │ • Exact Match / F1│   >= 99.5%        │ • Speed > 25 tok/s│ • 0 Safety Violations  │
│ • PPL < 3.5       │ • Human Eval Pass │ • 0 Stray Tabs    │ • VRAM within SLA │ • No General Drift     │
└───────────────────┴───────────────────┴───────────────────┴───────────────────┴────────────────────────┘
```

---

### Gate 1: Technical Convergence & Generalization Criteria

1. **Validation Loss Plateau:**
   * $\text{Eval Loss} \le 0.65$ without diverging from Training Loss ($\Delta(\text{Eval} - \text{Train}) \le 0.15$).
2. **Perplexity Threshold:**
   * $\text{Perplexity} = e^{\text{loss}} \le 3.50$ on domain evaluation test splits.
3. **No Catastrophic Memorization:**
   * Training loss must not drop below $0.05$ unless exact memorization is explicitly required (e.g. memorizing fixed legal statutes).

---

### Gate 2: Task-Specific Accuracy & Qualitative Criteria

Depending on your model's target use case:

| Use Case | Core Metric | Acceptance Threshold | Measurement Tool / Method |
| :--- | :--- | :--- | :--- |
| **Conversational Assistant** | Win Rate vs Base Model | $\ge 75\%$ Preference | LLM-as-a-Judge (GPT-4 / Claude Rubric) |
| **JSON / Structured Output** | Schema Parse Rate | $\ge 99.5\%$ Valid JSON | `pydantic` / `json.loads` automated parser |
| **Code Generation** | Pass@1 Execution Rate | $\ge 65\%$ Unit Test Pass | Sandbox code execution engine |
| **Classification / Routing** | F1-Score / Accuracy | $\ge 92.0\%$ F1 | Scikit-learn test set evaluation |
| **RAG / Fact Extraction** | Faithfulness & Groundedness | $\ge 95.0\%$ Grounded | Ragas / TruLens automated eval |

---

### Gate 3: Structural & Formatting Compliance

1. **Turn Tag Integrity:**
   * $100\%$ compliance with chat template delimiters (`<start_of_turn>`, `<end_of_turn>`).
   * Zero unclosed tags or hallucinated system prompts.
2. **Special Character Cleanliness:**
   * Zero mid-word tab (`\t`) drops or runaway space tokens.
   * Proper termination with `<eos>` (no infinite token generation loops).
3. **Non-Empty Response Guarantee:**
   * $0\%$ blank or single-EOS outputs on the validation benchmark.

---

### Gate 4: Latency, Throughput & Hardware SLA Criteria

1. **Time to First Token (TTFT):**
   * $\text{TTFT} \le 350\text{ ms}$ on target hardware.
2. **Token Generation Throughput:**
   * $\ge 25 \text{ tokens/second}$ per user stream on single GPU.
3. **VRAM Envelope:**
   * Maximum VRAM consumption (Model + KV Cache for max context) must not exceed $90\%$ of total GPU VRAM (e.g. $\le 7.2\text{GB}$ on an 8GB GPU).

---

### Gate 5: Safety, Alignment & Anti-Regression (Catastrophic Forgetting)

1. **Safety & Toxicity:**
   * $0\%$ toxic, harmful, or policy-violating outputs on standard red-teaming test suites.
2. **Catastrophic Forgetting Benchmark:**
   * Baseline general knowledge (tested on standard MMLU or GSM8k subset) must not degrade by more than $5\%$ compared to the un-fine-tuned base model.

---

## 3. The Golden Evaluation Set Methodology

An automated **Golden Evaluation Set** is a curated dataset of 50–200 challenging prompts that represent the complete operational spectrum of your use case.

```
Golden Evaluation Suite (100 Prompts Total)
├── 40 Standard Representative Queries (Core domain competency)
├── 30 Edge Cases & Ambiguous Queries (Testing reasoning and disambiguation)
├── 20 Formatting Stress Tests (Complex JSON schemas, tables, Markdown)
└── 10 Adversarial / Out-of-Scope Prompts (Testing refusal and boundary compliance)
```

### Implementing Automated LLM-as-a-Judge Evaluation

```python
"""
Automated Evaluation Script using LLM-as-a-Judge
Compares Fine-Tuned Model vs Base Model against Golden Dataset
"""

import json
from transformers import pipeline

def evaluate_response(judge_pipeline, prompt, base_output, finetuned_output):
    judge_prompt = f"""
[Instruction]
You are an expert impartial judge evaluating two AI assistant responses.
User Prompt: {prompt}

Response A (Base Model):
{base_output}

Response B (Fine-Tuned Model):
{finetuned_output}

Evaluate which response is superior in terms of accuracy, tone, concise formatting, and completeness.
Output valid JSON in this exact format:
{{"winner": "A" or "B" or "Tie", "reason": "brief explanation"}}
"""
    result = judge_pipeline(judge_prompt, max_new_tokens=150)
    return json.loads(result[0]["generated_text"])
```

---

## 4. Failure Mode Diagnostic Playbook

When an acceptance gate fails during iteration, apply these targeted remedies:

```
┌──────────────────────────────────────────────┬────────────────────────────────────────────────────────┐
│ Observed Failure Mode                        │ Actionable Iteration Remedy                            │
├──────────────────────────────────────────────┼────────────────────────────────────────────────────────┤
│ Model fails JSON schema in 5% of runs        │ • Inject 200 high-complexity JSON samples with error tags│
│                                              │ • Use constrained decoding (Outlines / JSON schema)   │
├──────────────────────────────────────────────┼────────────────────────────────────────────────────────┤
│ Model repeats phrases infinitely             │ • Increase repetition_penalty (1.0 -> 1.05)            │
│                                              │ • Add negative examples with penalty in dataset        │
├──────────────────────────────────────────────┼────────────────────────────────────────────────────────┤
│ Model forgets how to do basic math/reasoning │ • Mix in 10% general instruction data (e.g. FineTome)   │
│                                              │ • Lower LoRA rank (r=64 -> r=32) to prevent drift     │
├──────────────────────────────────────────────┼────────────────────────────────────────────────────────┤
│ Validation loss climbs while train loss drops│ • Stop training earlier (reduce epochs from 3 to 2)    │
│                                              │ • Increase weight_decay to 0.02 (L2 Regularization)    │
├──────────────────────────────────────────────┼────────────────────────────────────────────────────────┤
│ Model hallucinates non-existent facts        │ • Do NOT increase epochs (causes memorization)         │
│                                              │ • Implement RAG (Retrieval-Augmented Generation)       │
└──────────────────────────────────────────────┴────────────────────────────────────────────────────────┘
```

---

## 5. Production Sign-Off Checklist (Go / No-Go Gate)

Before deploying your fine-tuned LoRA adapter (`gemma_4_e4b_lora`) to users:

- [ ] **Data Sanitization Verified:** Zero empty turns, no broken tags, clean formatting.
- [ ] **Convergence Verified:** 1–2 full epochs completed, eval loss verified, no over-fitting.
- [ ] **Golden Eval Passed:** Win rate $> 75\%$ over base model across 100 golden prompts.
- [ ] **Format Adherence Verified:** Schema validator passes $100\%$ of automated test cases.
- [ ] **Inference Benchmark Tested:** Average latency $< 350\text{ms}$ TTFT on production GPU.
- [ ] **Adapter Export Verified:** LoRA weights successfully loaded in standalone [`infer_chat.py`](file:///home/abhishek/projects/unsloth/infer_chat.py) and verified.
