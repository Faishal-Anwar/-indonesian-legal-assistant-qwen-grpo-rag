# 🤖 Indonesian Legal Chatbot — Fine-tuned LLM with RAG

An end-to-end **Legal Assistant Chatbot** built for corporate legal teams, powered by a **fine-tuned Large Language Model (LLM)** integrated with a **Retrieval-Augmented Generation (RAG)** system over Indonesian labor law documents.

| **Info** | **Detail** |
|---|---|
| **Author** | Faishal Anwar Hasyim |
| **Email** | anwarfaishal86@gmail.com |

---

## 📋 Overview

This project consists of 3 main notebooks that build a legal chatbot system from scratch:

1. **SFT Fine-tuning** — Fine-tune Qwen2.5-1.5B using QLoRA on an Indonesian instruction dataset
2. **GRPO Training** — Train the model to perform structured reasoning with `<think>...</think>` tags using reward-based reinforcement learning
3. **RAG Pipeline** — Build a full retrieval system over 4 Indonesian labor law documents with advanced features (HyDE, Reranking, Fallback Search)

---

## 📁 Project Structure

```
├── Fine_tuning_submission_PGABL_Faishal_Anwar.ipynb   # Notebook 1: SFT Fine-tuning
├── GRPO_submission_PGABL_Faishal_Anwar.ipynb          # Notebook 2: GRPO Training
├── RAG_submission_PGABL_Faishal_Anwar.ipynb           # Notebook 3: RAG Pipeline
├── link_huggingface.txt                                # HuggingFace model links
├── requirements.txt                                    # Python dependencies
└── README.md                                           # Documentation
```

---

## 📓 Notebook Details

### 1. SFT Fine-tuning

Fine-tuning **Qwen2.5-1.5B** using **QLoRA 4-bit** quantization on `Ichsan2895/alpaca-gpt4-indonesian` dataset (49,969 samples).

- **Chat Template**: ChatML (`<|im_start|>`, `<|im_end|>`)
- **LoRA Targets**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Training**: 800 steps × 2 independent experiments with hyperparameter variations
- **Experiment 1**: lr=2e-4, linear scheduler → Loss 1.2896
- **Experiment 2**: lr=5e-5, cosine scheduler → Loss 1.3390

### 2. GRPO Training (Reinforcement Learning)

Train the model with **4 custom reward functions** for 200 steps using Group Relative Policy Optimization:

| Function | Description |
|---|---|
| `format_reward_func` | Graduated reward based on presence of `<think>`, `</think>` tags and final answer (max +1.0) |
| `reasoning_length_reward` | Reward by **character count** of reasoning: <50 → +0.2, 50-199 → +0.5, ≥200 → +1.0 |
| `correctness_reward` | ROUGE-L with **threshold 0.5**: score ≥0.5 → flat +1.0, below → 0.0 |
| `language_reward_func` | Reward based on Indonesian language usage detection |

The trained model produces structured output with `<think>...</think>` tags containing its reasoning process before the final answer.

### 3. RAG Pipeline

A complete RAG pipeline integrating the fine-tuned model with 4 Indonesian labor law documents (1,949 pages total).

**Law Documents:**
- PP No. 35/2021 — Fixed-term Employment (PKWT), Outsourcing, Working Hours, Overtime, Termination
- UU No. 6/2023 — Job Creation Law (Omnibus Law)
- PP No. 5/2021 — Risk-Based Business Licensing
- PP No. 51/2023 — Amendment to PP 35/2021

**RAG Features:**

| Feature | Description |
|---|---|
| **Parent-Child Chunking** | Parent (2000 chars) + Child (500 chars) for high-precision retrieval |
| **Ensemble Retriever** | FAISS Semantic (60%) + BM25 Keyword (40%), 5,337 chunks |
| **HyDE** | LLM dynamically generates 2 hypothetical documents per query |
| **Cross-Encoder Reranker** | ms-marco-MiniLM-L-6-v2 for Top-K reranking |
| **DuckDuckGo Fallback** | Automatic web search when relevance score falls below threshold |
| **Metadata Filtering** | Filter by document category and law name |
| **Source Citations** | Answers include document source references |
| **Interface** | Interactive Python Loop + Gradio Web UI |

---

## 🔗 Models on Hugging Face

| Model | Link |
|---|---|
| **SFT** | [Faishal-Anwar/qwen2.5-1.5b-pgabl-legal-sft-faishal](https://huggingface.co/Faishal-Anwar/qwen2.5-1.5b-pgabl-legal-sft-faishal) |
| **GRPO** | [Faishal-Anwar/qwen2.5-1.5b-pgabl-legal-grpo-faishal](https://huggingface.co/Faishal-Anwar/qwen2.5-1.5b-pgabl-legal-grpo-faishal) |

---

## 🚀 Getting Started

### Prerequisites
- Google Colab with **GPU runtime** (T4 or higher)
- Hugging Face account with access token
- Weights & Biases account (optional, for experiment tracking)

### Steps

1. Upload notebooks to Google Colab
2. Enable GPU runtime: `Runtime → Change runtime type → GPU`
3. Run notebooks in order:
   - `Fine_tuning_submission_PGABL_Faishal_Anwar.ipynb` (first)
   - `GRPO_submission_PGABL_Faishal_Anwar.ipynb` (second)
   - `RAG_submission_PGABL_Faishal_Anwar.ipynb` (third)
4. Enter your Hugging Face token when prompted
5. Law PDF documents will be automatically downloaded from Google Drive

---

## 💬 Example Questions

```
1. What is the maximum overtime hours per day and per week?
2. What compensation must a company provide to terminated employees?
3. What is the maximum duration for a fixed-term employment contract (PKWT)?
4. What are the provisions for annual leave and maternity leave for female workers?
5. What are the latest income tax (PPh 21) rules for employees earning above 10 million?
```

> Question 5 is outside the document scope — it triggers the **DuckDuckGo fallback search** feature.

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| **LLM** | Qwen2.5-1.5B |
| **Fine-tuning** | Unsloth + QLoRA + SFTTrainer |
| **RL** | GRPO (trl library) |
| **Embedding** | sentence-transformers/all-MiniLM-L6-v2 |
| **Vector DB** | FAISS |
| **Retrieval** | LangChain Ensemble (FAISS + BM25) |
| **Reranker** | Cross-Encoder ms-marco-MiniLM-L-6-v2 |
| **Query Transform** | HyDE (Hypothetical Document Embeddings) |
| **Web Fallback** | DuckDuckGo Search |
| **UI** | Gradio |
| **Tracking** | Weights & Biases |
| **Platform** | Google Colab (GPU) |

---

## 📄 License

This project was built for educational purposes.

