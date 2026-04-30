# BIS Standards Recommendation Engine

An advanced Retrieval-Augmented Generation (RAG) system designed to automate Bureau of Indian Standards (BIS) compliance discovery for Micro and Small Enterprises (MSEs). 

This project was built for the BIS Standards Hackathon and features a highly optimized RAG architecture specifically tuned for processing the **BIS SP 21** Building Materials dataset.

## 🚀 Key Innovations

1. **Standard-Aware Chunking:** 
   Traditional RAG arbitrarily splits documents. Our engine parses the PDF with custom regex to detect the "IS Number" and "Title", injecting this metadata into every chunk. This ensures isolated paragraphs always retain their regulatory context.
2. **Hybrid Retrieval (RRF):** 
   Combines Dense Vector Search (`all-MiniLM-L6-v2`) for semantic intent with Sparse Keyword Search (`BM25`) for exact technical matches, fused via Reciprocal Rank Fusion.
3. **Cross-Encoder Reranking:**
   Utilizes `FlashRank` (`ms-marco-TinyBERT-L-2-v2`) to compute pairwise attention across the top 15 candidates, maximizing the MRR @5 metric by forcing the exact standard to Rank #1.
4. **Anti-Hallucination Gate:**
   Strict JSON output generation via Ollama (`llama3.2`), coupled with an internal validation script that ensures recommended IS numbers actually exist in the retrieved context.

## 🛠️ Setup Instructions

### 1. Requirements
- Python 3.9+
- Ollama (Optional, but required for local LLM inference)

### 2. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/your-username/bis_standard_engine.git
cd bis_standard_engine
pip install -r requirements.txt
```

### 3. Add Dataset
Place the required hackathon dataset inside the `data/` folder:
- `data/bis_sp_21.pdf`

## 📊 Running Inference

The engine runs via the mandatory `inference.py` script. It takes an input JSON of product queries and outputs strict JSON recommendations.

```bash
python inference.py --input data/public_test_set.json --output data/team_results.json
```

## 🧪 Evaluation

To test the system against the hackathon constraints (Hit Rate @3, MRR @5, and Latency < 5s), run the evaluation script:

```bash
python eval_script.py
```
*(Note: If Ollama or the PDF is missing, the engine will gracefully fall back to mock data to demonstrate pipeline execution).*
