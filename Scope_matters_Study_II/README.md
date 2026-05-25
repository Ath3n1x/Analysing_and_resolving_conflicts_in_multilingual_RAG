# Scope Matters: Hallucination-Aware Evaluation of Retrieval Strategies for Multilingual RAG

Code for the CLEF 2026 paper:

> **Scope Matters: A Hallucination-Aware Evaluation of Retrieval Strategies for Multilingual RAG**  
> Submitted to CLEF 2026 — Conference Track, Experimental Part.

---

## Overview

This repository evaluates three retrieval corpus scopes across two multilingual QA benchmarks:

| Scope | Description |
|-------|-------------|
| **Mono** | Query-language passages only |
| **Cross** | Query-language + English passages (FAISS index) |
| **Full** | All language passages pooled |

**Generators** (4): Aya-23-8B · Qwen2.5-7B-Instruct · LLaMA-3-8B-Instruct · EuroLLM-9B-Instruct  
**Retrievers** (2): BGE-M3 · multilingual-E5-large  
**Datasets**: MLQA (7 languages) · XQuAD (11 languages)  
**Metrics**: EM · F1 · RHR · LMR · HALL · RUR · BERTScore · HP-F1 · LBS

---

## Repository Structure

```
multilingual-rag-clef2026/
├── src/
│   ├── chunker.py        # Language-aware passage chunking
│   ├── retriever.py      # BGE-M3 / E5 retrieval + FAISS scope indexes
│   ├── generator.py      # 4 generators, language-specific prompting
│   ├── metrics.py        # All metrics including HP-F1, LBS, RUR
│   ├── router.py         # Scope routing table + pretty-print
│   └── plotting.py       # 3 paper figures
├── experiments/
│   ├── run_mlqa.py       # MLQA entry point
│   └── run_xquad.py      # XQuAD entry point
├── configs/
│   ├── mlqa.yaml         # All hyperparameters for MLQA
│   └── xquad.yaml        # All hyperparameters for XQuAD
├── results/              # Auto-created; .csv/.json + figures/
├── requirements.txt
├── environment.yml
└── .env.example
```

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/your-org/multilingual-rag-clef2026.git
cd multilingual-rag-clef2026

# Conda (recommended)
conda env create -f environment.yml
conda activate mlrag

# Or pip
pip install -r requirements.txt
```

### 2. HuggingFace token

Models require a HuggingFace token with access to gated repos (LLaMA-3).

```bash
cp .env.example .env
# Edit .env and set HF_TOKEN=hf_your_token_here
export HF_TOKEN="hf_your_token_here"
```

---

## Running Experiments

### Quick smoke test (5 samples, one pair)

```bash
python experiments/run_mlqa.py \
  --generators aya --retrievers bgem3 \
  --langs en hi --samples 5 --no-plot
```

### Full MLQA run

```bash
python experiments/run_mlqa.py
```

### Full XQuAD run

```bash
python experiments/run_xquad.py
```

### Subset of generators / retrievers

```bash
# Only Aya and Qwen with both retrievers
python experiments/run_mlqa.py --generators aya qwen

# Only BGE-M3
python experiments/run_xquad.py --retrievers bgem3

# 4-bit quantisation (saves ~8 GB VRAM per model)
python experiments/run_mlqa.py --load-4bit
```

### CLI flags

| Flag | Description |
|------|-------------|
| `--config PATH` | Override config file (default: `configs/mlqa.yaml`) |
| `--generators ID [ID ...]` | Subset of generator IDs (aya, qwen, llama, eurollm) |
| `--retrievers ID [ID ...]` | Subset of retriever IDs (bgem3, e5) |
| `--langs CODE [...]` | Language subset |
| `--samples N` | QA pairs per language |
| `--load-4bit` | Load generators in 4-bit (requires bitsandbytes) |
| `--no-plot` | Skip matplotlib figure generation |

---

## Output

Each `(generator, retriever)` pair produces a subdirectory:

```
results/mlqa/aya_bgem3/
    results.csv          # Per-language metrics + routing decision
    results.json         # Full nested results for all scopes
    figures/
        aya_bgem3_answer_quality.png
        aya_bgem3_retrieval_quality.png
        aya_bgem3_decision_logic.png

results/mlqa/summary.json   # Routing table across all pairs
```

---

## Key Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| **RHR** | — | Retrieval Hit Rate: gold span in retrieved chunks |
| **LMR** | — | Language Match Rate: Unicode-range character fraction |
| **HALL** | — | Hallucination: F1 < 0.3 and context overlap < 0.5 |
| **RUR** | \|P∩C\|/\|P\| | Retrieval Utility Rate: predicted tokens in context |
| **HP-F1** | F1 × (1 − HALL/100) | Hallucination-penalised F1 |
| **LBS** | HP-F1_scope − HP-F1_mono | Language Benefit Score (routing signal) |

Routing decision: widest scope with `LBS > threshold` and `HALL < disqualify%`; else **mono**.

---

## Hardware Requirements

| Configuration | VRAM |
|---------------|------|
| fp16 (default) | ~16 GB per generator |
| 4-bit (`--load-4bit`) | ~8 GB per generator |
| BGE-M3 + E5 embeddings | ~4 GB additional |

All four generators are loaded and released sequentially; only one resides in GPU memory at a time.

---

## Citation

