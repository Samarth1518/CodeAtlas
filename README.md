# CodeAtlas 🗺️

> **AI-Powered Codebase Intelligence Platform** — Analyze any public GitHub repository, understand its architecture, explore semantic search, and query your codebase using strictly grounded Retrieval-Augmented Generation (RAG).

---

## 🌟 Key Features

1. **Repository Metadata & Tree Exploration (Phases 1 & 2)**
   - Instant validation of public GitHub repository URLs.
   - Fetches repository metadata (owner, branch, stars, primary language, visibility).
   - Recursive file and directory tree retrieval via GitHub's Git Trees API with client-side filtering.

2. **Safe Source Content Fetching (Phase 3)**
   - Security-first content retrieval with deterministic allowlist of text extensions (`.py`, `.ts`, `.tsx`, `.js`, `.jsx`, `.json`, `.yaml`, `.md`, `.rs`, `.go`, `.java`, etc.).
   - Unconditional filtering of secrets (`.env`, private keys, certificates, credentials) and heavy binary/media files.
   - Per-file size limits (512 KB) to prevent memory exhaustion.

3. **Intelligent Code Chunking (Phase 4)**
   - Structure-aware chunking for Python (AST) and JavaScript/TypeScript (heuristic function/class boundaries).
   - Line-range preservation (`L1–L42`), character counts, and unique deterministic chunk IDs.

4. **Local Persistent Embeddings & Semantic Search (Phase 5)**
   - Local dense vector embeddings via `sentence-transformers` (`all-MiniLM-L6-v2`).
   - High-performance, zero-external-dependency cosine similarity search powered by NumPy.
   - Persistent on-disk vector store per repository (`storage/vector_store/`).

5. **AI Codebase Assistant via Grounded RAG (Phase 6)**
   - Retrieval-Augmented Generation (RAG) answering developer questions with exact line citations.
   - Zero hallucination guarantee: prompt mandates strict grounding in retrieved repository chunks only.
   - OpenAI-compatible provider abstraction (works seamlessly with Google Gemini, OpenAI, OpenRouter, Groq, Mistral, LocalAI).
   - Graceful 503 fallback when `LLM_API_KEY` is not configured.

6. **Repository Architecture & Dependency Insights (Phase 7)**
   - 100% deterministic analysis of directory roles, technology stacks, frameworks, CI/CD pipelines, and documentation.
   - Multi-ecosystem manifest parsing: `package.json`, `requirements.txt`, `pyproject.toml`, `Cargo.toml`, `go.mod`, `Gemfile`, `composer.json`.
   - Evidence-based architectural observations without invented facts.

7. **Production-Ready UI & Developer Experience (Phase 8)**
   - Modern dark-mode aesthetic with smooth scrolling, quick-jump navigation, and interactive code viewers.
   - Centralized API configuration supporting custom domains and containerized deployments.
   - 250+ unit and integration tests across all pipeline layers.

---

## 🏗️ Architecture Overview

```
GitHub Repository URL
         │
         ▼
[ Flask API Gateway (app.py) ]
         │
         ├── POST /api/repos/analyze       ──► GitHub REST API (metadata + recursive tree)
         ├── POST /api/contents/fetch      ──► Safe File Filter (allowlist & secret protection)
         ├── POST /api/chunks/generate     ──► AST & Semantic Chunker
         ├── POST /api/index/build         ──► sentence-transformers (all-MiniLM-L6-v2)
         │                                       │
         │                                       ▼
         │                                 [ Local Vector Store (NumPy) ]
         │                                       ▲
         ├── POST /api/search              ──────┤ (Cosine Similarity Search)
         ├── POST /api/insights/analyze    ──► Deterministic Architecture Engine
         └── POST /api/chat                ──► RAG Pipeline ──► LLM Provider (Gemini / OpenAI)
                                                                 │
                                                                 ▼
                                                        Grounded Answer + Citations
```

---

## 💻 Tech Stack

| Layer | Technology | Purpose |
| :--- | :--- | :--- |
| **Frontend** | React 18, TypeScript, Vite | Fast, responsive Single Page Application |
| **Styling** | Vanilla CSS (Modern design system) | Glassmorphism, tailored gradients, responsive grids |
| **Backend** | Python 3.10+, Flask, Flask-CORS | Modular REST API with blueprint architecture |
| **Embeddings** | `sentence-transformers` (`all-MiniLM-L6-v2`) | Local 384-dimensional vector embeddings |
| **Vector Store** | NumPy | In-memory & on-disk cosine similarity indexing |
| **LLM Provider** | `openai` Python SDK (Universal interface) | Google Gemini, OpenAI, OpenRouter, Groq |
| **Security** | Custom regex & file classification engine | Strict credential stripping and binary exclusion |

---

## 🚀 Quick Start Guide

### Prerequisites
- **Node.js** 18+ and `npm`
- **Python** 3.10+ (with `venv` support)
- *(Optional)* A **GitHub Personal Access Token** (increases GitHub API rate limit from 60 to 5,000 req/hr)
- *(Optional)* A **Gemini or OpenAI API key** (enables the AI Codebase Assistant)

---

### 1. Backend Setup

```bash
cd backend

# 1. Create and activate a Python virtual environment
# Windows (PowerShell):
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux:
python3 -m venv venv
source venv/bin/activate

# 2. Install backend dependencies
pip install -r requirements.txt

# 3. Configure environment variables
cp .env.example .env
```

Edit `backend/.env` with your desired configuration:

```env
# Flask configuration
FLASK_ENV=development
FLASK_DEBUG=1
PORT=5000

# GitHub API Token (Optional but recommended for public repo rate limits)
GITHUB_TOKEN=your_github_personal_access_token_here

# LLM Configuration (Optional: required for AI Q&A)
# For Google Gemini (Recommended):
LLM_API_KEY=your_gemini_api_key_here
LLM_MODEL=gemini-2.5-flash
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai

# For OpenAI:
# LLM_API_KEY=sk-proj-...
# LLM_MODEL=gpt-4o-mini
# LLM_BASE_URL=https://api.openai.com/v1
```

```bash
# 4. Start the backend server
python app.py
```
> Backend runs at **http://localhost:5000** (Health check: `GET http://localhost:5000/api/health`)

---

### 2. Frontend Setup

In a new terminal:

```bash
cd frontend

# 1. Install dependencies
npm install

# 2. (Optional) Configure environment
cp .env.example .env

# 3. Start development server
npm run dev
```
> Frontend runs at **http://localhost:5173**

---

## 📡 API Endpoints Overview

All endpoints return JSON and accept standard payloads.

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Service health status check |
| `POST` | `/api/repos/analyze` | Validate public repository and retrieve metadata + file tree |
| `POST` | `/api/contents/fetch` | Safely retrieve decoded source file contents |
| `POST` | `/api/chunks/generate` | Chunk source files into structured code blocks |
| `POST` | `/api/index/build` | Generate embeddings and persist local vector index |
| `POST` | `/api/search` | Execute semantic code search against vector index |
| `POST` | `/api/insights/analyze` | Extract deterministic tech stack, dependencies & architecture |
| `POST` | `/api/chat` | Query codebase using grounded RAG pipeline |

---

## 🔒 Security & Privacy Guarantees

- **No Secret Leakage**: `GITHUB_TOKEN` and `LLM_API_KEY` are never returned in API responses, printed in console logs, or exposed in error messages.
- **Sensitive File Exclusion**: `.env`, `.env.*`, `id_rsa`, `*.pem`, `*.key`, and credential files are unconditionally ignored and cannot be fetched.
- **Strict Grounding**: The AI assistant answers queries exclusively using retrieved code chunks and is forbidden from guessing unverified APIs or dependencies.
- **Repository Isolation**: Semantic search and vector indices are strictly segregated by repository URL key.
- **Public-Only Access**: CodeAtlas enforces public repository restrictions to prevent unauthorized data exposure.

---

## 🧪 Testing

CodeAtlas includes a comprehensive automated test suite covering unit tests, integration tests, security filtering, and full end-to-end workflows.

### Running Backend Tests

```bash
cd backend
python -m unittest discover -s tests -v
```

### Running Frontend Type-Check & Production Build

```bash
cd frontend
npm run build
```

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
