# FastAPI Gemini RAG Application

A Retrieval-Augmented Generation (RAG) API built with **FastAPI**, **LangChain**, **ChromaDB**, and **Google Gemini** (`gemini-2.0-flash` for chat and `models/text-embedding-004` for embeddings).

## Project Structure

- `ingest.py` — Loads a PDF (`DOC_PATH`, default `resume.pdf`), splits text into chunks (`chunk_size=500`, `chunk_overlap=50`), generates Gemini embeddings, and persists them to `./chroma_db` in collection `"docs"`.
- `app.py` — FastAPI server exposing `GET /health` and `POST /ask` using similarity search (`k=3`) and Gemini `gemini-2.0-flash`.
- `requirements.txt` — Pinned Python dependencies.
- `.env.example` — Environment variable template (`GOOGLE_API_KEY`, `DOC_PATH`).

## Setup

1. Create and activate a virtual environment:

   **Windows (PowerShell):**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   **macOS / Linux:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment variables:

   **Windows (PowerShell):**
   ```powershell
   Copy-Item .env.example .env
   ```

   **macOS / Linux:**
   ```bash
   cp .env.example .env
   ```

   Edit `.env` and set your Google Gemini API key and PDF path:
   ```env
   GOOGLE_API_KEY=your_actual_google_api_key
   DOC_PATH=resume.pdf
   ```

4. Place your PDF file at the `DOC_PATH` location (for example, `resume.pdf` in the project root).

## Run

1. **Ingest the PDF into ChromaDB:**
   ```bash
   python ingest.py
   ```
   Expected output:
   ```text
   Stored <N> chunks in Chroma collection 'docs' at './chroma_db'.
   ```

2. **Start the FastAPI server:**
   ```bash
   uvicorn app:app --reload
   ```
   The API will be available at `http://127.0.0.1:8000` (interactive docs at `http://127.0.0.1:8000/docs`).

## Test Endpoints

- **Health check:**
  ```bash
  curl http://127.0.0.1:8000/health
  ```
  Response:
  ```json
  {"status":"ok"}
  ```

- **Ask a question:**
  ```bash
  curl -X POST http://127.0.0.1:8000/ask \
    -H "Content-Type: application/json" \
    -d '{"question": "What programming languages are listed?"}'
  ```
  Response:
  ```json
  {
    "answer": "...",
    "sources": ["chunk 1...", "chunk 2...", "chunk 3..."],
    "latency_ms": 420
  }
  ```
