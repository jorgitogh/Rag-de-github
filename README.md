# RAG de GitHub

Streamlit app to clone a public GitHub repository, index its codebase, and query it with RAG.

## Stack

- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (local)
- Vector store: `FAISS`
- LLM: Groq API with fixed model `llama-3.3-70b-versatile`
- UI: Streamlit

## Features

- Shallow clone of public GitHub repos
- File filtering to skip binaries and heavy folders
- Line-aware chunks with source citations (`file:start-end`)
- Q&A over repo context
- Patch generation as unified diff text

## Files

- `app.py`: main application (indexing, retrieval, generation, UI)
- `requirements.txt`: dependencies
- `runtime.txt`: Streamlit Cloud runtime (`python-3.11`)

## Requirements

- Python 3.11
- `GROQ_API_KEY`

## Local Run

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Optional env var:

```bash
export GROQ_API_KEY="your_key_here"  # PowerShell: $env:GROQ_API_KEY="your_key_here"
```

## Usage

1. Open the app.
2. In the sidebar, set repo URL, optional branch, embedding model, and `GROQ_API_KEY`.
3. Click `Clonar + Indexar`.
4. Use `Chat RAG` to ask questions or `Generar patch (diff)` for code changes.

## Main Config (`app.py`)

- `MAX_FILES = 600`
- `MAX_FILE_BYTES = 700_000`
- `CHUNK_SIZE = 1200`
- `CHUNK_OVERLAP = 150`
- `MAX_CHUNKS = 1200`
- `TOP_K = 6`

These values affect indexing speed, memory use, and retrieval quality.

## Streamlit Cloud Notes

- Keep `runtime.txt` and `requirements.txt` committed.
- If deployment is stuck in "in the oven", check logs, then `Clear cache` and `Reboot app`.

## Troubleshooting

- `Missing GROQ_API_KEY`: set it in sidebar or as env var.
- Slow indexing: use smaller repos, reduce chunk/file limits, or simplify embedding model.
- Build issues on Cloud: verify dependency pins and reboot after pushing changes.
