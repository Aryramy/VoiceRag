import os
import asyncio
import uuid
from typing import List, Dict, Any, Tuple
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_openai import OpenAIEmbeddings
from app.config import settings

STORAGE_PATH = os.path.abspath(settings.QDRANT_STORAGE_PATH)
COLLECTION_NAME = settings.QDRANT_COLLECTION_NAME

_client: AsyncQdrantClient | None = None
_client_init_failed = False

def get_qdrant_client() -> AsyncQdrantClient | None:
    global _client, _client_init_failed
    if _client is None and not _client_init_failed:
        try:
            os.makedirs(STORAGE_PATH, exist_ok=True)
            _client = AsyncQdrantClient(path=STORAGE_PATH)
        except Exception:
            # Another process (e.g. FastAPI backend) holds the exclusive disk lock
            _client_init_failed = True
            return None
    return _client

client = get_qdrant_client()

def get_embeddings() -> OpenAIEmbeddings:
    """Always instantiates with current event loop to prevent 'Event loop is closed' across threads/requests."""
    return OpenAIEmbeddings(
        model=settings.effective_embedding_model,
        openai_api_base=settings.effective_embedding_base_url,
        openai_api_key=settings.effective_embedding_api_key,
        check_embedding_ctx_length=False
    )

async def init_qdrant() -> None:
    """Initialize the local Qdrant collection if it does not exist."""
    q_client = get_qdrant_client()
    if q_client is None:
        return
    collections = await q_client.get_collections()
    names = [c.name for c in collections.collections]
    if COLLECTION_NAME not in names:
        await q_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
        )

async def search_qdrant(query: str, top_k: int = 3, min_score_threshold: float = 0.70) -> Tuple[List[Dict[str, Any]], float]:
    """Search the local embedded Qdrant collection for relevant documents with fallback to internal HTTP if locked."""
    q_client = get_qdrant_client()
    
    # If this process cannot acquire local Qdrant lock, proxy search to running FastAPI backend
    if q_client is None:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=10.0) as http_c:
                res = await http_c.post(
                    "http://127.0.0.1:8000/api/internal-search",
                    json={"query": query, "top_k": top_k, "min_score_threshold": min_score_threshold}
                )
                if res.status_code == 200:
                    data = res.json()
                    return data.get("docs", []), float(data.get("max_score", 0.0))
        except Exception as e:
            print(f"[WARN] Internal search proxy failed: {e}")
            return [], 0.0

    await init_qdrant()
    embeddings = get_embeddings()
    vector = await embeddings.aembed_query(query)
    
    search_result = None
    for attempt in range(3):
        try:
            search_result = await q_client.query_points(
                collection_name=COLLECTION_NAME,
                query=vector,
                limit=top_k
            )
            break
        except Exception as e:
            if attempt == 2:
                raise e
            await asyncio.sleep(0.5)
    
    points = search_result.points if hasattr(search_result, "points") else search_result
    if not points:
        return [], 0.0
        
    max_score = max([hit.score for hit in points])
    docs = [
        {
            "id": f"chunk_{hit.id}",
            "text": hit.payload["text"],
            "score": hit.score,
            "source_file": hit.payload.get("source_file", hit.payload.get("source", "Knowledge Base")),
            "chunk_index": hit.payload.get("chunk_index", 0)
        }
        for hit in points
        if hit.score >= min_score_threshold and hit.payload and "text" in hit.payload
    ]
    
    return docs, float(max_score)

async def ingest_documents(texts: List[str], metadatas: List[Dict[str, Any]] | None = None) -> int:
    """Embed and store text chunks in local embedded Qdrant."""
    q_client = get_qdrant_client()
    if q_client is None:
        raise RuntimeError("Cannot ingest documents: local Qdrant is locked or unavailable in this process.")

    await init_qdrant()
    embeddings = get_embeddings()

    if not texts:
        return 0

    vectors = await embeddings.aembed_documents(texts)
    points = []
    for i, (text, vector) in enumerate(zip(texts, vectors)):
        point_id = str(uuid.uuid4())
        payload = {"text": text}
        if metadatas and i < len(metadatas):
            payload.update(metadatas[i])
            if "source" in payload and "source_file" not in payload:
                payload["source_file"] = payload["source"]
        points.append(
            PointStruct(
                id=point_id,
                vector=vector,
                payload=payload
            )
        )

    await q_client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )
    return len(points)
