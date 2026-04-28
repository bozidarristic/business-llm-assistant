from pathlib import Path
import chromadb


class ChromaVectorStore:
    def __init__(self, persist_dir: str, collection_name: str):
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def reset(self) -> None:
        name = self.collection.name
        try:
            self.client.delete_collection(name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(name=name)

    def add(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ) -> None:
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def count(self) -> int:
        return self.collection.count()

    def list_sources(self) -> list[dict]:
        result = self.collection.get(include=["metadatas"])
        metadatas = result.get("metadatas") or []
        by_source: dict[str, dict] = {}

        for metadata in metadatas:
            source = metadata.get("source", "unknown")
            entry = by_source.setdefault(
                source,
                {
                    "source": source,
                    "type": metadata.get("type", "unknown"),
                    "chunks": 0,
                },
            )
            entry["chunks"] += 1

        return sorted(by_source.values(), key=lambda item: item["source"])

    def query(
        self,
        query_embedding: list[float],
        top_k: int,
        where: dict | None = None,
    ) -> dict:
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
