
# Fine-Tuning Mistral with LoRA for Insurance Conversation Summarization

## Overview

This project demonstrates parameter-efficient fine-tuning of the **MistralForCausalLM** model using **LoRA (Low-Rank Adaptation)** for domain-specific summarization of insurance sales conversations.

The objective is to transform fragmented enterprise communication (emails, CRM notes, transcripts) into concise, structured, and actionable summaries tailored for sales professionals.

The notebook serves as both:
- A working fine-tuning demonstration
- A reusable blueprint for domain-adaptive LLM training across industries

---

## Model Architecture

The base model used is **MistralForCausalLM**, which consists of the following key components:

### 1. Embedding Layer
- Converts input tokens into dense representations
- Embedding dimension: **4096**
- Vocabulary size: **32,000 tokens**

### 2. Decoder Stack (32 Layers)
Each `MistralDecoderLayer` includes:

**Self-Attention Mechanism**
- Query, Key, Value, and Output projections
- Rotary positional embeddings
- 4-bit precision support for memory efficiency

**Feedforward Network (MLP)**
- Expands dimensionality to **14,336**
- Projects back to **4096**
- Uses **SiLU activation**

**Normalization**
- Input normalization
- Post-attention normalization
- Implemented using `MistralRMSNorm`

### 3. Final Normalization Layer
Ensures stable output representations before projection.

### 4. Linear Output Head
- Projects 4096-dimension outputs back to vocabulary space (32,000 tokens)
- Enables next-token prediction for generation

---

## LoRA Integration

LoRA adapters are attached to specific projection layers during instantiation:

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`
- `gate_proj`
- `up_proj`
- `down_proj`

This enables training only a small fraction of parameters (~0.5%) while keeping the base model frozen, significantly reducing compute and memory requirements.

---

## Training Strategy

The fine-tuning process incorporates established deep learning best practices:

- Low learning rate for stable parameter updates
- Early stopping based on validation loss (negative log likelihood)
- Gradient checkpointing for memory efficiency
- 8-bit optimizer for reduced GPU memory usage

### Demonstration Mode (60 Steps)

For demonstration purposes, training is limited to **60 steps** instead of a full epoch.

- A **step** = one batch update  
- An **epoch** = one full pass over the dataset  

Limiting to steps allows:
- Quick experimentation  
- Fast iteration  
- Reduced compute time in educational settings  

To perform a full training run, uncomment:

```python
num_train_epochs = 1
```

---

## Evaluation Results

We observed a significant improvement in **BERTScore** after fine-tuning.

| Model Version        | BERTScore |
|----------------------|-----------|
| Before Fine-Tuning   | 0.16      |
| After Fine-Tuning    | 0.53      |

### Interpretation

- The base model produced summaries with limited domain alignment and structural consistency.
- After LoRA fine-tuning on insurance-specific dialogue data, the model achieved a **231% relative improvement** in BERTScore.
- Outputs became more concise, context-aware, and aligned with enterprise communication needs.

This improvement demonstrates the effectiveness of lightweight, domain-adaptive fine-tuning for enterprise summarization tasks.

---

## Repository Structure

```
project_root/
│
├── Finetuning_with_LLMs.ipynb
├── artifacts/
│   └── finetuned_mistral_lora/
├── README.md
```

---

## Technical Stack

- PyTorch  
- Hugging Face Transformers  
- TRL (Supervised Fine-Tuning)  
- Unsloth (Efficient GPU training)  
- PEFT (LoRA)  
- BERTScore for evaluation  

---

## Generalization

Although demonstrated in the insurance domain, this architecture can be adapted for:

- Healthcare documentation summarization  
- Legal transcript condensation  
- Financial advisory reporting  
- Customer support conversation analysis  
- Enterprise account intelligence  

The workflow provides a scalable template for domain-adaptive summarization systems.

---

## Conclusion

This project illustrates how parameter-efficient fine-tuning can significantly enhance domain-specific summarization performance while maintaining computational efficiency.

By combining Mistral's architecture with LoRA adapters and structured training practices, we create a lightweight yet powerful summarization model tailored for enterprise communication.
