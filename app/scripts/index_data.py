from pathlib import Path

from app.core.config import settings
from app.core.logging import get_logger
from app.ingestion.loaders import load_all_documents
from app.indexing.chunking import chunk_documents
from app.indexing.embeddings import LocalEmbedder
from app.indexing.vectorstore import ChromaVectorStore

logger = get_logger(__name__)


def main() -> None:
    logger.info("Loading raw business documents...")
    documents = load_all_documents(Path("data/raw"))

    logger.info("Chunking documents...")
    texts, metadatas, ids = chunk_documents(
        documents=documents,
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
    )

    logger.info("Embedding chunks locally...")
    embedder = LocalEmbedder(settings.embedding_model_name)
    embeddings = embedder.embed_texts(texts)

    logger.info("Creating local Chroma vector store...")
    vectorstore = ChromaVectorStore(
        persist_dir=settings.chroma_persist_dir,
        collection_name=settings.chroma_collection_name,
    )

    logger.info("Resetting collection and adding new chunks...")
    vectorstore.reset()
    vectorstore.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    logger.info("Indexing completed. Indexed %s chunks.", len(ids))


if __name__ == "__main__":
    main()
