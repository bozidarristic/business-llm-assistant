from app.core.logging import get_logger
from app.services.rag_service import rebuild_index

logger = get_logger(__name__)


def main() -> None:
    logger.info("Rebuilding local vector index...")
    indexed_chunks = rebuild_index()
    logger.info("Indexing completed. Indexed %s chunks.", indexed_chunks)


if __name__ == "__main__":
    main()
