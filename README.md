# Semantic Query Engine

Chat with your PDFs using fast semantic retrieval and LLM reasoning. Upload documents, ask natural‑language questions, and get clean, grounded answers in a modern, responsive UI.

> Industrial frontend, simple backend, configurable providers.

## ✨ Features

- Semantic PDF Q&A (RAG) with sentence‑aware chunking
- Vector search (cosine) using 768‑dim embeddings
- Refined answers (markdown + syntax highlighting + copy buttons)
- Sleek UI: drag‑and‑drop upload, status toasts, clean chat bubbles
- Dashboard metric cards (processed PDFs, questions, latency, success)
- FastAPI backend; Pinecone for vectors; Gemini for LLM (configurable)

## 🧱 Architecture

1. PDF upload → text extraction (PyPDF2) → sentence‑aware chunking (spaCy)
2. Embeddings → upsert to vector index (Pinecone)
3. Question → embed → vector search (top‑k)
4. LLM synthesizes a concise, helpful answer grounded in retrieved chunks
5. Frontend renders markdown with code highlighting and copy actions

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- API keys: Pinecone + Gemini (or your preferred LLM)

### Install

```bash
# from project root
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
# source venv/bin/activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Configure

Create a `.env` in the project root:

```env
GEMINI_API_KEY=your_gemini_key
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=your_index_name
```

### Run

```bash
python main.py
# open http://127.0.0.1:8000
```

## 🖥️ Frontend

- Card‑based dashboard (metric cards only)
- Upload section with button + drag & drop
- Professional chat UI (markdown, highlighting, copy, toasts)

## 🔧 Configuration Notes

- Embedding model: `models/embedding-001` (Gemini). Swap as needed.
- Index metric: cosine; embedding dimension: 768.
- Top‑k is adaptive based on question complexity.

## 📸 Screenshots 

> Add screenshots here (dashboard, upload, chat).

## 🧪 Tester

You can post to the ask endpoint:

```bash
curl -X POST -d "question=What does this document cover?" http://127.0.0.1:8000/api/ask
```

## 🛡️ Security

- Env keys are loaded via `python-dotenv`. Do not commit `.env`.
- Frontend sanitizes markdown with DOMPurify.

## 🛠️ Tech Stack

- Backend: FastAPI, PyPDF2, spaCy, python‑dotenv
- Vector DB: Pinecone
- LLM: Gemini (configurable)
- Frontend: Bootstrap + vanilla JS, Marked, DOMPurify, highlight.js

## 🧭 Roadmap

- Streaming responses
- Citations with snippet previews
- Multi‑file corpus management

## 📄 License

No license specified yet. Consider adding MIT/Apache‑2.0 for open collaboration.

---

