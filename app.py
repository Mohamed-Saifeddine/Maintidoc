from __future__ import annotations

import hashlib
from pathlib import Path

import streamlit as st

from maintidoc.agent import DocumentAgent
from maintidoc.config import settings
from maintidoc.health import ollama_health
from maintidoc.indexing import DocumentIndexer
from maintidoc.vector_store import QdrantVectorStore

st.set_page_config(page_title="MAINTiDOC", page_icon="📚", layout="wide")
settings.ensure_directories()


@st.cache_resource
def get_indexer() -> DocumentIndexer:
    return DocumentIndexer(settings)


@st.cache_resource
def get_agent() -> DocumentAgent:
    return DocumentAgent(settings)


def save_upload(uploaded_file) -> Path:
    data = uploaded_file.getvalue()
    digest = hashlib.sha256(data).hexdigest()[:12]
    safe_name = Path(uploaded_file.name).name.replace(" ", "_")
    destination = settings.uploads_dir / f"{digest}_{safe_name}"
    destination.write_bytes(data)
    return destination


def run_indexing(pdf_path: Path) -> None:
    progress_bar = st.progress(0.0)
    status = st.empty()

    def update(message: str, value: float) -> None:
        progress_bar.progress(value)
        status.info(message)

    try:
        stats = get_indexer().index(pdf_path, update)
    except Exception as exc:
        status.error(f"Échec de l'indexation : {exc}")
        st.exception(exc)
        return
    status.success("Document indexé avec succès.")
    st.session_state["last_stats"] = stats.model_dump()
    get_agent.clear()


st.title("📚 MAINTiDOC")
st.caption("RAG multimodal agentique — texte, tableaux et images")

with st.sidebar:
    st.header("État du système")
    ollama_ok, installed_models = ollama_health(settings)
    st.write("Ollama", "✅" if ollama_ok else "❌")
    required = [settings.embed_model, settings.vision_model, settings.agent_model]
    for model in required:
        installed = model in installed_models or model.split(":")[0] in {
            item.split(":")[0] for item in installed_models
        }
        st.write(f"{'✅' if installed else '⚠️'} `{model}`")
    try:
        counts = QdrantVectorStore(settings).counts()
        st.write(f"Textes/tableaux : **{counts['text_and_tables']}**")
        st.write(f"Images : **{counts['images']}**")
    except Exception as exc:
        st.warning(f"Qdrant indisponible : {exc}")

    st.divider()
    st.caption(f"Embeddings : {settings.embed_model}")
    st.caption(f"Vision : {settings.vision_model}")
    st.caption(f"Agent : {settings.agent_model}")

index_tab, chat_tab, architecture_tab = st.tabs(["1. Indexer", "2. Interroger", "Architecture"])

with index_tab:
    st.subheader("Choisir et indexer un PDF")
    uploaded = st.file_uploader("Importer un document PDF", type=["pdf"])
    col1, col2 = st.columns(2)
    with col1:
        if uploaded and st.button(
            "Indexer le PDF importé", type="primary", use_container_width=True
        ):
            run_indexing(save_upload(uploaded))
    with col2:
        if st.button("Indexer le rapport UNICEF fourni", use_container_width=True):
            if settings.default_pdf.exists():
                run_indexing(settings.default_pdf)
            else:
                st.error(f"Document absent : {settings.default_pdf}")

    if "last_stats" in st.session_state:
        stats = st.session_state["last_stats"]
        st.success(
            f"{stats['pages']} pages — {stats['text_chunks']} passages — "
            f"{stats['tables']} tableaux — {stats['images']} images"
        )

with chat_tab:
    st.subheader("Poser une question")
    st.info(
        "L'agent choisit lui-même entre la recherche textuelle, visuelle ou multimodale. "
        "Les réponses doivent citer la page et l'élément utilisés."
    )
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ex. Quelles sont les principales dépenses du programme ?")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            progress_status = st.status(
                "L'agent recherche les sources puis prépare la réponse…",
                expanded=True,
            )
            try:
                result = get_agent().ask(question, progress=progress_status.write)
                progress_status.update(
                    label="Réponse terminée",
                    state="complete",
                    expanded=False,
                )
                st.markdown(result.answer)
                if result.tools_used:
                    st.caption("Outils utilisés : " + ", ".join(result.tools_used))
                if result.sources:
                    with st.expander(f"Sources récupérées ({len(result.sources)})"):
                        for source in result.sources:
                            st.markdown(f"**{source.citation}** — score {source.score:.3f}")
                            st.write(source.content[:700])
                            if source.image_path and Path(source.image_path).exists():
                                st.image(source.image_path, width=320)
                st.session_state.messages.append({"role": "assistant", "content": result.answer})
            except Exception as exc:
                progress_status.update(
                    label="La génération a échoué",
                    state="error",
                    expanded=True,
                )
                st.error(f"Impossible de répondre : {exc}")
                st.exception(exc)

with architecture_tab:
    st.subheader("Pipeline exécuté")
    st.code(
        """PDF
 ├─ LlamaIndex + PyMuPDF → passages textuels
 ├─ pdfplumber → tableaux Markdown
 └─ PyMuPDF → images originales
       ├─ BGE-M3/Ollama → vecteurs texte et tableaux (1024)
       └─ CLIP → vecteurs images (512)
                    ↓
            Qdrant (2 collections)
                    ↓
     Agent LangChain + Qwen/Ollama
       ├─ outil recherche texte/tableaux
       ├─ outil recherche images
       └─ outil recherche multimodale
                    ↓
       Réponse sourcée dans Streamlit""",
        language="text",
    )
