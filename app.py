# app.py
import os
import re
import shutil
import hashlib
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional

import streamlit as st
from git import Repo
from groq import Groq

from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2" 
DEFAULT_LLM_MODEL = "llama-3.1-8b-instant"
MAX_FILES = 600
MAX_FILE_BYTES = 700_000  
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150
MAX_CHUNKS = 1200
TOP_K = 6

IGNORE_DIRS = {
    ".git", ".github", ".venv", "venv", "env", "__pycache__",
    "node_modules", "dist", "build", ".next", ".cache",
    ".idea", ".vscode", "coverage", ".pytest_cache",
    "target", "out"
}

ALLOWED_EXT = {
    ".py", ".ipynb", ".js", ".ts", ".tsx", ".jsx",
    ".java", ".go", ".rs", ".cpp", ".c", ".h", ".hpp",
    ".cs", ".php", ".rb", ".swift", ".kt",
    ".sql", ".md", ".txt", ".yml", ".yaml", ".toml", ".json",
    ".ini", ".cfg", ".env", ".sh", ".bash", ".zsh",
    ".dockerfile", "Dockerfile"
}

BINARY_EXT = {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip", ".tar", ".gz", ".7z", ".exe", ".bin"}


def is_github_repo_url(url: str) -> bool:
    return bool(re.match(r"^https?://github\.com/[^/]+/[^/]+/?$", url.strip()))

def safe_repo_dir(url: str) -> str:
    h = hashlib.sha1(url.strip().encode("utf-8")).hexdigest()[:10]
    return f"repo_{h}"

def should_skip_path(p: Path) -> bool:
    parts = set(p.parts)
    if parts & IGNORE_DIRS:
        return True
    if p.is_dir():
        return True
    ext = p.suffix.lower()
    name = p.name
    if ext in BINARY_EXT:
        return True
    if name == "package-lock.json" or name == "yarn.lock" or name == "pnpm-lock.yaml":
        return True
    if name == "Dockerfile":
        return False
    if name.lower().endswith("dockerfile"):
        return False
    if ext and ext in ALLOWED_EXT:
        return False
    if name in {".env", ".gitignore"}:
        return False
    return True

def read_text_file(path: Path) -> Optional[str]:
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            return None
        try:
            return path.read_text(encoding="utf-8", errors="strict")
        except UnicodeDecodeError:
            return path.read_text(encoding="latin-1", errors="ignore")
    except Exception:
        return None

def make_line_aware_docs(repo_root: Path) -> List[Document]:
    """
    Crea documentos preservando líneas para poder citar (archivo + rango de líneas).
    Estrategia: trocear por líneas en bloques de ~N caracteres, guardando start_line/end_line.
    """
    docs: List[Document] = []
    files = []
    for p in repo_root.rglob("*"):
        if should_skip_path(p):
            continue
        files.append(p)
        if len(files) >= MAX_FILES:
            break

    for fp in files:
        text = read_text_file(fp)
        if not text:
            continue

        rel = str(fp.relative_to(repo_root))
        lines = text.splitlines()
        buf = []
        buf_len = 0
        start_line = 1
        current_line = 1

        def flush(end_line: int):
            nonlocal buf, buf_len, start_line
            if not buf:
                return
            content = "\n".join(buf)
            docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "path": rel,
                        "start_line": start_line,
                        "end_line": end_line,
                    },
                )
            )
            buf, buf_len = [], 0
            start_line = end_line + 1

        for line in lines:
            line_len = len(line) + 1
            if buf_len + line_len > CHUNK_SIZE * 2 and buf: 
                flush(current_line - 1)
            buf.append(line)
            buf_len += line_len
            current_line += 1

        flush(current_line - 1)

    return docs

@st.cache_resource(show_spinner=False)
def get_embeddings(model_name: str, hf_api_key: str):
    return HuggingFaceInferenceAPIEmbeddings(
        api_key=hf_api_key,
        model_name=model_name,
    )

@st.cache_resource(show_spinner=False)
def get_groq_client(api_key: str):
    return Groq(api_key=api_key)

def groq_generate(client: Groq, model_name: str, prompt: str) -> str:
    completion = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=500,
    )
    return (completion.choices[0].message.content or "").strip()

def build_vectorstore(docs: List[Document], embedding_model: str, hf_api_key: str, repo_url: str) -> Chroma:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    embeddings = get_embeddings(embedding_model, hf_api_key)
    collection_name = f"repo_{hashlib.sha1(repo_url.encode('utf-8')).hexdigest()[:12]}"
    persist_dir = Path(tempfile.gettempdir()) / "github_rag_chroma"
    persist_dir.mkdir(parents=True, exist_ok=True)
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=str(persist_dir),
    )

def format_sources(docs: List[Document], max_sources: int = 5) -> str:
    seen = set()
    out = []
    for d in docs:
        p = d.metadata.get("path", "unknown")
        sl = d.metadata.get("start_line", "?")
        el = d.metadata.get("end_line", "?")
        key = (p, sl, el)
        if key in seen:
            continue
        seen.add(key)
        out.append(f"- `{p}:{sl}-{el}`")
        if len(out) >= max_sources:
            break
    return "\n".join(out) if out else "_(sin fuentes)_"

def rag_answer(query: str, vs: Chroma, llm_client: Groq, llm_model: str, k: int = TOP_K) -> Tuple[str, List[Document]]:
    retriever = vs.as_retriever(search_kwargs={"k": k})
    ctx_docs = retriever.get_relevant_documents(query)

    context = "\n\n".join(
        [f"[{i+1}] FILE={d.metadata.get('path')} LINES={d.metadata.get('start_line')}-{d.metadata.get('end_line')}\n{d.page_content}"
         for i, d in enumerate(ctx_docs)]
    )

    system = (
        "Eres un asistente experto en ingeniería de software. Responde SOLO usando el CONTEXTO.\n"
        "Si no está en el contexto, dilo claramente ('no lo veo en el repo').\n"
        "Cuando afirmes algo, apóyalo en evidencias del contexto.\n"
        "Devuelve una respuesta clara, con bullets si ayuda.\n"
    )

    prompt = f"{system}\n\nCONTEXTO:\n{context}\n\nPREGUNTA:\n{query}\n\nRESPUESTA:"
    answer = groq_generate(llm_client, llm_model, prompt)
    return answer, ctx_docs

def generate_patch(request: str, vs: Chroma, llm_client: Groq, llm_model: str, k: int = TOP_K) -> Tuple[str, List[Document]]:
    """
    Genera un diff estilo git. No aplica cambios; solo muestra patch.
    """
    retriever = vs.as_retriever(search_kwargs={"k": k})
    ctx_docs = retriever.get_relevant_documents(request)

    context = "\n\n".join(
        [f"[{i+1}] FILE={d.metadata.get('path')} LINES={d.metadata.get('start_line')}-{d.metadata.get('end_line')}\n{d.page_content}"
         for i, d in enumerate(ctx_docs)]
    )

    system = (
        "Eres un senior engineer. Genera UN PATCH en formato unified diff (git diff) basado en el contexto.\n"
        "Reglas:\n"
        "- Solo modifica archivos que existan en el contexto.\n"
        "- Si no puedes, explica por qué.\n"
        "- Devuelve SOLO el diff, sin texto extra.\n"
    )
    prompt = f"{system}\n\nCONTEXTO:\n{context}\n\nPETICIÓN:\n{request}\n\nDIFF:"
    diff = groq_generate(llm_client, llm_model, prompt)
    return diff, ctx_docs


st.set_page_config(page_title="GitHub RAG (LangChain)", layout="wide")
st.title("🔎 GitHub RAG — pregunta a cualquier repo (con citas)")

with st.sidebar:
    st.header("Indexación")
    repo_url = st.text_input("URL repo público", placeholder="https://github.com/user/repo")
    branch = st.text_input("Branch (opcional)", placeholder="main")
    embedding_model = st.text_input("Embeddings", value=DEFAULT_EMBEDDING_MODEL)
    hf_api_key = st.text_input("HUGGINGFACE_API_KEY", value=os.getenv("HUGGINGFACE_API_KEY", ""), type="password")
    llm_model = st.text_input("LLM (Groq)", value=DEFAULT_LLM_MODEL)
    groq_api_key = st.text_input("GROQ_API_KEY", value=os.getenv("GROQ_API_KEY", ""), type="password")
    top_k = st.slider("Top-K chunks", 3, 12, TOP_K)
    do_index = st.button("📥 Clonar + Indexar", type="primary")

if "vs" not in st.session_state:
    st.session_state.vs = None
if "repo_path" not in st.session_state:
    st.session_state.repo_path = None
if "llm" not in st.session_state:
    st.session_state.llm = None
if "llm_model" not in st.session_state:
    st.session_state.llm_model = DEFAULT_LLM_MODEL

if do_index:
    if not is_github_repo_url(repo_url):
        st.error("URL inválida. Usa formato: https://github.com/user/repo")
    elif not hf_api_key.strip():
        st.error("Missing HUGGINGFACE_API_KEY. Add it in the sidebar or as an environment variable.")
    elif not groq_api_key.strip():
        st.error("Missing GROQ_API_KEY. Add it in the sidebar or as an environment variable.")
    else:
        with st.spinner("Clonando repo…"):
            base_tmp = Path(tempfile.gettempdir())
            target = base_tmp / safe_repo_dir(repo_url)
            if target.exists():
                shutil.rmtree(target, ignore_errors=True)

            # shallow clone
            if branch.strip():
                Repo.clone_from(repo_url, target, depth=1, branch=branch.strip())
            else:
                Repo.clone_from(repo_url, target, depth=1)

            st.session_state.repo_path = str(target)

        with st.spinner("Leyendo archivos y creando documentos…"):
            docs = make_line_aware_docs(Path(st.session_state.repo_path))
            st.write(f"📄 Documentos base: {len(docs)}")

        with st.spinner("Construyendo índice vectorial (ChromaDB)…"):
            vs = build_vectorstore(docs, embedding_model, hf_api_key.strip(), repo_url.strip())
            st.session_state.vs = vs

        with st.spinner("Conectando con Groq..."):
            st.session_state.llm = get_groq_client(groq_api_key.strip())
            st.session_state.llm_model = llm_model

        st.success("✅ Repo indexado. Ya puedes preguntar.")

col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("💬 Chat RAG")
    q = st.text_area("Pregunta", placeholder="¿Cómo funciona el pipeline de ingestion? ¿Dónde está el entrypoint?")
    if st.button("Responder", disabled=(st.session_state.vs is None or st.session_state.llm is None)):
        with st.spinner("Pensando…"):
            ans, src = rag_answer(q, st.session_state.vs, st.session_state.llm, st.session_state.llm_model, k=top_k)
        st.markdown(ans)
        st.markdown("**Fuentes**")
        st.markdown(format_sources(src))

with col2:
    st.subheader("🩹 Generar patch (diff)")
    req = st.text_area("Petición de cambio", placeholder="Añade validación de input y maneja errores en app.py")
    if st.button("Generar diff", disabled=(st.session_state.vs is None or st.session_state.llm is None)):
        with st.spinner("Generando…"):
            diff, src = generate_patch(req, st.session_state.vs, st.session_state.llm, st.session_state.llm_model, k=top_k)
        st.code(diff, language="diff")
        st.markdown("**Fuentes usadas**")
        st.markdown(format_sources(src))
