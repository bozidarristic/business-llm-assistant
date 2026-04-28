from app.data.schema import RetrievedChunk
from app.indexing.embeddings import LocalEmbedder
from app.indexing.vectorstore import ChromaVectorStore


class BusinessRetriever:
    def __init__(
        self,
        embedder: LocalEmbedder,
        vectorstore: ChromaVectorStore,
        top_k: int,
    ):
        self.embedder = embedder
        self.vectorstore = vectorstore
        self.top_k = top_k

    def retrieve(self, query: str, where: dict | None = None) -> list[RetrievedChunk]:
        query_embedding = self.embedder.embed_query(query)
        result = self.vectorstore.query(
            query_embedding=query_embedding,
            top_k=self.top_k,
            where=where,
        )

        chunks = []
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        for chunk_id, content, metadata, distance in zip(ids, docs, metas, distances):
            chunks.append(
                RetrievedChunk(
                    id=chunk_id,
                    content=content,
                    metadata=metadata,
                    score=distance,
                )
            )

        return chunks
