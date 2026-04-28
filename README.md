# Business LLM Assistant - Local RAG

A small LLM-powered assistant for business data: customers, leads, support tickets, and internal documents.

The project provides a practical local retrieval-augmented generation pipeline using:

- local generation with Hugging Face Transformers by default
- optional Ollama support
- retrieval-augmented generation
- structured CSV records plus unstructured Markdown policies
- grounded question answering
- summarization and customer-support response drafting

## What this project does

The assistant can answer questions such as:

- "What is the current status of Acme Retail?"
- "Summarize all open support tickets for customer CUST-001."
- "Which leads should sales prioritize?"
- "Draft a professional reply to Beta Manufacturing about the SLA issue."
- "What does the refund policy say?"

The system answers using only the provided business dataset.

## Architecture

```text
data/raw/
    customers.csv
    leads.csv
    support_tickets.csv
    internal_docs.md

        -> ingestion

BusinessDocument objects with metadata

        -> chunking

short text chunks

        -> embeddings

SentenceTransformers local embedding model

        -> vector store

ChromaDB persistent local database

        -> retrieval + light structured guardrails

top-k relevant chunks, focused by company when possible

        -> generation

local LLM through Transformers or Ollama
```

## Setup

Create and activate the environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The default `.env` uses:

```env
LLM_BACKEND=transformers
LLM_MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
```

This keeps the default setup lightweight and does not require Ollama. If Ollama is available, set:

```env
LLM_BACKEND=ollama
OLLAMA_MODEL=mistral
OLLAMA_BASE_URL=http://localhost:11434
```

## Run indexing

```bash
python -m app.scripts.index_data
```

## Ask questions

```bash
python -m app.scripts.ask "What is the status of Acme Retail?"
```

```bash
python -m app.scripts.ask "Which leads should sales prioritize?"
```

```bash
python -m app.scripts.ask "Draft a professional reply to Beta Manufacturing about the SLA issue."
```

## Run the UI

```bash
streamlit run app/ui/streamlit_app.py
```

The UI supports:

- uploading CSV, Markdown, and text files
- rebuilding the local vector index
- asking questions through a simple chat-style form
- viewing the available files and indexed sources

## Design Notes

The assistant is designed as a modular local RAG system. Business records are converted into normalized text documents with metadata, chunked, embedded locally, and stored in Chroma. At query time, the system retrieves relevant context and passes it to a local model with a strict grounded prompt.

Because the dataset contains structured business records, the assistant also includes a small guardrail layer for exact workflows such as lead prioritization and SLA reply drafting. This prevents a small local model from mixing up critical facts like priority, customer, plan, or sender. The LLM remains useful for general grounded answers, while deterministic handling protects high-risk business responses.

## Possible improvements

- hybrid retrieval with BM25 plus embeddings
- reranking
- source citations in a UI
- FastAPI or Streamlit interface
- incremental indexing
- document deletion
- role-based access control
- evaluation dataset for groundedness and relevance
