from pathlib import Path

from app.core.config import settings
from app.data.schema import BusinessDocument
from app.generation.assistant import BusinessAssistant
from app.generation.local_transformers_client import LocalTransformersClient
from app.generation.ollama_client import OllamaClient
from app.indexing.chunking import chunk_documents
from app.indexing.embeddings import LocalEmbedder
from app.indexing.vectorstore import ChromaVectorStore
from app.ingestion.loaders import load_all_documents
from app.retrieval.retriever import BusinessRetriever


RAW_DATA_DIR = Path("data/raw")
UPLOAD_DIR = Path("data/uploads")


def build_embedder() -> LocalEmbedder:
    return LocalEmbedder(settings.embedding_model_name)


def build_vectorstore() -> ChromaVectorStore:
    return ChromaVectorStore(
        persist_dir=settings.chroma_persist_dir,
        collection_name=settings.chroma_collection_name,
    )


def build_llm():
    if settings.llm_backend == "ollama":
        return OllamaClient(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
        )
    if settings.llm_backend == "transformers":
        return LocalTransformersClient(
            model_name=settings.llm_model_name,
            max_new_tokens=settings.llm_max_new_tokens,
        )
    raise ValueError("Unsupported LLM_BACKEND. Use 'transformers' or 'ollama'.")


def build_assistant(
    embedder: LocalEmbedder | None = None,
    vectorstore: ChromaVectorStore | None = None,
    llm=None,
) -> BusinessAssistant:
    embedder = embedder or build_embedder()
    vectorstore = vectorstore or build_vectorstore()
    retriever = BusinessRetriever(
        embedder=embedder,
        vectorstore=vectorstore,
        top_k=settings.top_k,
    )
    return BusinessAssistant(retriever=retriever, llm=llm or build_llm())


def load_documents() -> list[BusinessDocument]:
    return load_all_documents(raw_data_dir=RAW_DATA_DIR, upload_dir=UPLOAD_DIR)


def rebuild_index(
    embedder: LocalEmbedder | None = None,
    vectorstore: ChromaVectorStore | None = None,
) -> int:
    documents = load_documents()
    texts, metadatas, ids = chunk_documents(
        documents=documents,
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
    )

    embedder = embedder or build_embedder()
    embeddings = embedder.embed_texts(texts)

    vectorstore = vectorstore or build_vectorstore()
    vectorstore.reset()
    vectorstore.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    return len(ids)


def save_uploaded_file(file_name: str, content: bytes) -> Path:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = Path(file_name).name
    target = UPLOAD_DIR / safe_name
    target.write_bytes(content)
    return target


def list_available_files() -> list[dict]:
    files = []
    for category, directory in [("sample", RAW_DATA_DIR), ("uploaded", UPLOAD_DIR)]:
        if not directory.exists():
            continue
        for path in sorted(directory.iterdir()):
            if path.is_file() and path.name != ".gitkeep":
                files.append(
                    {
                        "name": path.name,
                        "category": category,
                        "size_kb": round(path.stat().st_size / 1024, 1),
                    }
                )
    return files
