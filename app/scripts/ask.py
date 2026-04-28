import sys

from rich.console import Console
from rich.panel import Panel

from app.core.config import settings
from app.generation.assistant import BusinessAssistant
from app.generation.local_transformers_client import LocalTransformersClient
from app.generation.ollama_client import OllamaClient
from app.indexing.embeddings import LocalEmbedder
from app.indexing.vectorstore import ChromaVectorStore
from app.retrieval.retriever import BusinessRetriever

console = Console()


def build_assistant() -> BusinessAssistant:
    embedder = LocalEmbedder(settings.embedding_model_name)
    vectorstore = ChromaVectorStore(
        persist_dir=settings.chroma_persist_dir,
        collection_name=settings.chroma_collection_name,
    )
    retriever = BusinessRetriever(
        embedder=embedder,
        vectorstore=vectorstore,
        top_k=settings.top_k,
    )
    if settings.llm_backend == "ollama":
        llm = OllamaClient(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
        )
    elif settings.llm_backend == "transformers":
        llm = LocalTransformersClient(
            model_name=settings.llm_model_name,
            max_new_tokens=settings.llm_max_new_tokens,
        )
    else:
        raise ValueError(
            "Unsupported LLM_BACKEND. Use 'transformers' or 'ollama'."
        )

    return BusinessAssistant(retriever=retriever, llm=llm)


def main() -> None:
    if len(sys.argv) < 2:
        console.print('Usage: python -m app.scripts.ask "Your question here"')
        raise SystemExit(1)

    question = " ".join(sys.argv[1:])
    assistant = build_assistant()
    answer = assistant.answer(question)

    console.print(Panel(question, title="Question"))
    console.print(Panel(answer, title="Assistant Answer"))


if __name__ == "__main__":
    main()
