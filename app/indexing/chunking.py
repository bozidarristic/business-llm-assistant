from app.data.schema import BusinessDocument


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def chunk_documents(
    documents: list[BusinessDocument],
    chunk_size: int,
    overlap: int,
) -> tuple[list[str], list[dict], list[str]]:
    texts = []
    metadatas = []
    ids = []

    for doc in documents:
        chunks = chunk_text(doc.content, chunk_size=chunk_size, overlap=overlap)

        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc.id}-chunk-{i}"
            metadata = {
                **doc.metadata,
                "source": doc.source,
                "document_id": doc.id,
                "chunk_index": i,
            }

            texts.append(chunk)
            metadatas.append(metadata)
            ids.append(chunk_id)

    return texts, metadatas, ids
