# CodeLens — AI Codebase Intelligence Engine

> **RAG-Powered Semantic Code Search & AI Chat for Any GitHub Repository**

CodeLens lets you **paste any public GitHub repo URL**, indexes the entire codebase into a vector database, and then lets you **ask questions in plain English** — getting precise, context-aware answers grounded in the actual source code.

![CodeLens Demo](https://img.shields.io/badge/Status-Live-brightgreen) ![Python](https://img.shields.io/badge/Python-3.10+-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688) ![Gemini](https://img.shields.io/badge/Google_Gemini-API-4285F4)

---

##  Features

| Feature | Description |
|---------|-------------|
| **RAG Pipeline** | Retrieval-Augmented Generation using FAISS vector search + Gemini LLM |
| **File Explorer** | Interactive file tree with extension-based icons and search filtering |
| **AI Chat** | Ask questions about any codebase in natural language |
| **Conversation Mode** | Multi-turn chat with sliding window context (last 10 messages) |
| **Context Sources** | See exactly which code chunks were used, with similarity scores |
| **Auto Summary** | Generates a 3-sentence codebase overview on indexing |
| **Model Fallback** | Automatically rotates through Gemini models if quota is hit |

---

## Architecture

```
User Question
     ↓
① EMBED the question (Gemini Embedding API)
     ↓
② IT retrieve's top-8 most similar code chunks (FAISS cosine similarity)
     ↓
③ AUGMENT the LLM prompt with retrieved chunks as context
     ↓
④ GENERATE answer using Gemini Flash
     ↓
Answer + Source Citations + Similarity Scores

```

---

## Tech Stack

- **Backend**: Python, FastAPI, Uvicorn
- **Vector DB**: FAISS (Facebook AI Similarity Search)
- **Embeddings**: Google Gemini Embedding API (`gemini-embedding-001`)
- **LLM**: Google Gemini Flash (with automatic model fallback)
- **Code Parser**: Go CLI for repository cloning, file discovery, file tree generation, and text chunking
- **Git**: System `git` command used by the parser for repository cloning
- **Frontend**: Vanilla HTML/CSS/JS with Marked.js + Highlight.js

---

## Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/upadhyay74aman/Code_Lens.git
cd Code_Lens
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Get a Gemini API key
- Go to [Google AI Studio](https://aistudio.google.com/apikey)
- Create a free API key

### 4. Create `.env` file
```
GEMINI_API_KEY=your_api_key_here
```

### 5. Run the server
```bash
python main.py
```
Or:
```bash
uvicorn main:app --port 8000
```

### 6. Open in browser
Navigate to **http://localhost:8000** and start exploring codebases!

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serves the frontend UI |
| `POST` | `/api/index` | Clone & index a GitHub repository |
| `POST` | `/api/query` | Single Q&A semantic search |
| `POST` | `/api/chat` | Multi-turn conversation mode |
| `GET` | `/api/filetree` | Get the indexed repo's file tree |
| `GET` | `/api/status` | Get indexing statistics |

---

## Project Structure

```
Code_Lens/
├── main.py           # FastAPI backend (RAG pipeline, FAISS, Gemini)
├── index.html        # Frontend UI (IDE-style dark theme)
├── parser/           # Go parser CLI for cloning, file discovery, and chunking
├── build.ps1         # Windows helper to test and build the parser binary
├── requirements.txt  # Python dependencies
├── .env              # API key (not committed)
├── .gitignore        # Git ignore rules
└── README.md         # This file
```

---

## How RAG Works in CodeLens

1. **Indexing Phase**: Repository files are split into ~1000 character chunks with 200 char overlap. Each chunk is embedded using Gemini's embedding model and stored in a FAISS Inner Product index.

2. **Query Phase**: Your question is embedded using the same model, then FAISS finds the top-8 most semantically similar code chunks via cosine similarity.

3. **Generation Phase**: The retrieved chunks are injected into a carefully crafted prompt, and Gemini Flash generates a grounded answer citing specific files and line numbers.

---

## License

MIT License — feel free to use, modify, and distribute.
---
