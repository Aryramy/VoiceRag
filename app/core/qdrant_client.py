import os
import uuid
from typing import List, Dict, Any, Tuple
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_openai import OpenAIEmbeddings
from app.config import settings

STORAGE_PATH = os.path.abspath(settings.QDRANT_STORAGE_PATH)
COLLECTION_NAME = settings.QDRANT_COLLECTION_NAME

_client: AsyncQdrantClient | None = None
_embeddings: OpenAIEmbeddings | None = None

def get_qdrant_client() -> AsyncQdrantClient:
    global _client
    if _client is None:
        os.makedirs(STORAGE_PATH, exist_ok=True)
        _client = AsyncQdrantClient(path=STORAGE_PATH)
    return _client

def get_embeddings() -> OpenAIEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(
            model=settings.OPENROUTER_EMBEDDING_MODEL,
            openai_api_base=settings.OPENROUTER_BASE_URL,
            openai_api_key=settings.OPENROUTER_API_KEY
        )
    return _embeddings

async def init_qdrant() -> None:
    """Initialize the local Qdrant collection if it does not exist."""
    client = get_qdrant_client()
    collections = await client.get_collections()
    names = [c.name for c in collections.collections]
    if COLLECTION_NAME not in names:
        await client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
        )

async def search_qdrant(query: str, top_k: int = 3, min_score_threshold: float = 0.70) -> Tuple[List[Dict[str, Any]], float]:
    """Search the local embedded Qdrant collection for relevant documents."""
    await init_qdrant()
    client = get_qdrant_client()
    embeddings = get_embeddings()
    
    vector = await embeddings.aembed_query(query)
    
    # Qdrant client async query
    search_result = await client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=top_k
    )
    
    points = search_result.points if hasattr(search_result, "points") else search_result
    if not points:
        return [], 0.0
        
    max_score = max([hit.score for hit in points])
    docs = [
        {"id": f"chunk_{hit.id}", "text": hit.payload["text"], "score": hit.score}
        for hit in points
        if hit.score >= min_score_threshold and hit.payload and "text" in hit.payload
    ]
    
    return docs, float(max_score)

async def ingest_documents(texts: List[str], metadatas: List[Dict[str, Any]] | None = None) -> int:
    """Embed and store text chunks in local embedded Qdrant."""
    await init_qdrant()
    client = get_qdrant_client()
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
        points.append(
            PointStruct(
                id=point_id,
                vector=vector,
                payload=payload
            )
        )

    await client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )
    return len(points)

