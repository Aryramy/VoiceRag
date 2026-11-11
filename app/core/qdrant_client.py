import os
import gc
import json
import pickle
import shutil
import sqlite3
import asyncio
import logging
import time
import uuid
import weakref
from typing import List, Dict, Any, Tuple
import httpx
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from app.config import settings

logger = logging.getLogger("qdrant_client")

STORAGE_PATH = os.path.abspath(settings.QDRANT_STORAGE_PATH)
COLLECTION_NAME = settings.QDRANT_COLLECTION_NAME
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384

_client: AsyncQdrantClient | None = None
_client_init_failed = False
_collection_initialized = False
_pending_recovered_points: list[tuple[Any, dict[str, Any]]] = []

# Loop-safe singleton caches for persistent HTTP keep-alive pools (worker -> FastAPI proxy)
_http_clients: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, httpx.AsyncClient]" = (
    weakref.WeakKeyDictionary()
)
_init_locks: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock]" = (
    weakref.WeakKeyDictionary()
)

# Module-level singleton for local offline FastEmbed ONNX embeddings (no network/loop binding)
_embeddings_client: FastEmbedEmbeddings | None = None


def _extract_sqlite_payloads(collection_name: str) -> list[tuple[Any, dict[str, Any]]]:
    """Reads stored payloads directly from local Qdrant SQLite if vector dimensions became inhomogeneous."""
    sqlite_path = os.path.join(STORAGE_PATH, "collection", collection_name, "storage.sqlite")
    if not os.path.exists(sqlite_path):
        return []
    recovered: list[tuple[Any, dict[str, Any]]] = []
    seen_texts: set[tuple[str, str]] = set()
    conn = sqlite3.connect(sqlite_path)
    try:
        rows = conn.execute("SELECT id, point FROM points").fetchall()
        for _, blob in rows:
            pt = pickle.loads(blob)
            payload = dict(getattr(pt, "payload", None) or {})
            text = str(payload.get("text", "")).strip()
            src = str(payload.get("source_file", payload.get("source", "")))
            if not text or (src, text) in seen_texts:
                continue
            seen_texts.add((src, text))
            recovered.append((pt.id, payload))
    finally:
        conn.close()
    return recovered


def _force_cleanup_collection_dir(q_client: AsyncQdrantClient, collection_name: str) -> None:
    """Closes Windows SQLite handle held by local Qdrant before removing the collection directory."""
    inner = getattr(q_client, "_client", None)
    cols = getattr(inner, "collections", None)
    if isinstance(cols, dict) and collection_name in cols:
        col_obj = cols.get(collection_name)
        if col_obj is not None and hasattr(col_obj, "close"):
            try:
                col_obj.close()
            except Exception:
                pass
    gc.collect()


def get_qdrant_client() -> AsyncQdrantClient | None:
    global _client, _client_init_failed, _pending_recovered_points
    if _client is None and not _client_init_failed:
        try:
            os.makedirs(STORAGE_PATH, exist_ok=True)
            _client = AsyncQdrantClient(path=STORAGE_PATH)
        except ValueError as e:
            # Recover gracefully if local storage.sqlite contains mixed vector dims after an unclosed Windows delete
            if "inhomogeneous shape" in str(e).lower():
                logger.warning(f"[MIGRATION][qdrant] Recovering mixed-dimension local storage: {e}")
                _pending_recovered_points = _extract_sqlite_payloads(COLLECTION_NAME)
                gc.collect()
                col_dir = os.path.join(STORAGE_PATH, "collection", COLLECTION_NAME)
                if os.path.exists(col_dir):
                    shutil.rmtree(col_dir, ignore_errors=True)
                meta_path = os.path.join(STORAGE_PATH, "meta.json")
                if os.path.exists(meta_path):
                    try:
                        with open(meta_path, "r", encoding="utf-8") as mf:
                            meta = json.load(mf)
                        meta.get("collections", {}).pop(COLLECTION_NAME, None)
                        with open(meta_path, "w", encoding="utf-8") as mf:
                            json.dump(meta, mf)
                    except Exception:
                        pass
                try:
                    _client = AsyncQdrantClient(path=STORAGE_PATH)
                    return _client
                except Exception as retry_err:
                    logger.warning(f"[WARN] Qdrant recovery retry failed: {retry_err}")
                    _client_init_failed = True
                    return None
            _client_init_failed = True
            return None
        except Exception:
            # Another process (e.g. FastAPI backend) holds the exclusive disk lock
            _client_init_failed = True
            return None
    return _client


client = get_qdrant_client()


def get_http_client() -> httpx.AsyncClient:
    """Returns a persistent keep-alive httpx.AsyncClient bound to the current event loop."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None:
        cached = _http_clients.get(loop)
        if cached is not None and not cached.is_closed:
            return cached

    http_c = httpx.AsyncClient(
        timeout=httpx.Timeout(10.0, connect=5.0),
        limits=httpx.Limits(max_keepalive_connections=10, keepalive_expiry=60.0),
    )
    if loop is not None:
        _http_clients[loop] = http_c
    return http_c


def get_embeddings() -> FastEmbedEmbeddings:
    """Returns a persistent module-level FastEmbedEmbeddings singleton for <20ms offline BGE embeddings."""
    global _embeddings_client
    if _embeddings_client is not None:
        return _embeddings_client

    t0 = time.perf_counter()
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    emb = FastEmbedEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    # Warm up ONNX runtime once so the very first voice turn incurs zero cold session allocation
    emb.embed_query("warmup")
    _embeddings_client = emb
    logger.info(
        f"[PERF][embeddings_init] Created FastEmbedEmbeddings({EMBEDDING_MODEL_NAME}) singleton in {(time.perf_counter() - t0)*1000:.2f}ms"
    )
    return _embeddings_client


async def _migrate_collection_dimension(q_client: AsyncQdrantClient) -> None:
    """Safely re-indexes existing points when migrating from 1536-dim cloud vectors to 384-dim FastEmbed vectors."""
    t_mig0 = time.perf_counter()
    saved_points: list[tuple[Any, dict[str, Any]]] = []
    offset = None

    while True:
        batch, next_offset = await q_client.scroll(
            collection_name=COLLECTION_NAME,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for pt in batch:
            payload = dict(pt.payload or {})
            text = str(payload.get("text", "")).strip()
            if text:
                saved_points.append((pt.id, payload))
        if next_offset is None:
            break
        offset = next_offset

    _force_cleanup_collection_dir(q_client, COLLECTION_NAME)
    await q_client.delete_collection(collection_name=COLLECTION_NAME)
    col_dir = os.path.join(STORAGE_PATH, "collection", COLLECTION_NAME)
    if os.path.exists(col_dir):
        gc.collect()
        shutil.rmtree(col_dir, ignore_errors=True)
    await q_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )

    if saved_points:
        embeddings = get_embeddings()
        texts = [payload["text"] for _, payload in saved_points]
        vectors = await embeddings.aembed_documents(texts)
        new_points = [
            PointStruct(id=pt_id, vector=vec, payload=payload)
            for (pt_id, payload), vec in zip(saved_points, vectors)
        ]
        await q_client.upsert(collection_name=COLLECTION_NAME, points=new_points)

    logger.info(
        f"[MIGRATION][qdrant] Migrated {len(saved_points)} points in '{COLLECTION_NAME}' "
        f"to dim={EMBEDDING_DIM} ({EMBEDDING_MODEL_NAME}) in {(time.perf_counter() - t_mig0)*1000:.1f}ms"
    )


async def init_qdrant() -> None:
    """Initialize local Qdrant collection, verify 384-dim FastEmbed alignment, and cache readiness."""
    global _collection_initialized, _pending_recovered_points
    if _collection_initialized:
        return

    q_client = get_qdrant_client()
    if q_client is None:
        return

    loop = asyncio.get_running_loop()
    lock = _init_locks.get(loop)
    if lock is None:
        lock = asyncio.Lock()
        _init_locks[loop] = lock

    async with lock:
        if _collection_initialized:
            return
        t0 = time.perf_counter()
        collections = await q_client.get_collections()
        names = [c.name for c in collections.collections]
        if COLLECTION_NAME not in names:
            await q_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )
            if _pending_recovered_points:
                embeddings = get_embeddings()
                texts = [payload["text"] for _, payload in _pending_recovered_points]
                vectors = await embeddings.aembed_documents(texts)
                new_points = [
                    PointStruct(id=pt_id, vector=vec, payload=payload)
                    for (pt_id, payload), vec in zip(_pending_recovered_points, vectors)
                ]
                await q_client.upsert(collection_name=COLLECTION_NAME, points=new_points)
                logger.info(
                    f"[MIGRATION][qdrant] Re-indexed {len(new_points)} recovered points into '{COLLECTION_NAME}' (dim={EMBEDDING_DIM})"
                )
                _pending_recovered_points = []
        else:
            info = await q_client.get_collection(COLLECTION_NAME)
            vectors_cfg = getattr(info.config.params, "vectors", None)
            existing_dim = getattr(vectors_cfg, "size", None)
            if existing_dim is not None and existing_dim != EMBEDDING_DIM:
                logger.warning(
                    f"[MIGRATION][qdrant] Dimension mismatch in '{COLLECTION_NAME}' "
                    f"(found {existing_dim}, expected {EMBEDDING_DIM}). Re-indexing with {EMBEDDING_MODEL_NAME}..."
                )
                await _migrate_collection_dimension(q_client)
        _collection_initialized = True
        logger.info(
            f"[PERF][qdrant_init] Verified collection '{COLLECTION_NAME}' (dim={EMBEDDING_DIM}) "
            f"in {(time.perf_counter() - t0)*1000:.2f}ms"
        )


async def search_qdrant(
    query: str,
    top_k: int = 3,
    min_score_threshold: float = settings.FALLBACK_SCORE_THRESHOLD,
) -> Tuple[List[Dict[str, Any]], float]:
    """Search the local embedded Qdrant collection for relevant documents with fallback to internal HTTP if locked."""
    t_total_start = time.perf_counter()
    q_client = get_qdrant_client()

    # If this process cannot acquire local Qdrant lock, proxy search via persistent keep-alive HTTP client
    if q_client is None:
        try:
            http_c = get_http_client()
            t_proxy0 = time.perf_counter()
            res = await http_c.post(
                "http://127.0.0.1:8000/api/internal-search",
                json={"query": query, "top_k": top_k, "min_score_threshold": min_score_threshold},
            )
            t_proxy_ms = (time.perf_counter() - t_proxy0) * 1000
            if res.status_code == 200:
                data = res.json()
                docs = data.get("docs", [])
                max_score = float(data.get("max_score", 0.0))
                logger.info(
                    f"[PERF][qdrant_proxy] total_http_ms={t_proxy_ms:.2f}ms status={res.status_code} "
                    f"hits={len(docs)} max_score={max_score:.3f}"
                )
                return docs, max_score
        except Exception as e:
            logger.warning(f"[WARN] Internal search proxy failed: {e}")
            return [], 0.0

    t_init0 = time.perf_counter()
    await init_qdrant()
    t_init_ms = (time.perf_counter() - t_init0) * 1000

    t_embed0 = time.perf_counter()
    embeddings = get_embeddings()
    vector = await embeddings.aembed_query(query)
    t_embed_ms = (time.perf_counter() - t_embed0) * 1000

    t_query0 = time.perf_counter()
    search_result = None
    for attempt in range(3):
        try:
            search_result = await q_client.query_points(
                collection_name=COLLECTION_NAME,
                query=vector,
                limit=top_k,
            )
            break
        except Exception as e:
            if attempt == 2:
                raise e
            await asyncio.sleep(0.5)
    t_query_ms = (time.perf_counter() - t_query0) * 1000

    points = search_result.points if hasattr(search_result, "points") else search_result
    if not points:
        logger.info(
            f"[PERF][qdrant_local] init_ms={t_init_ms:.2f}ms embed_ms={t_embed_ms:.1f}ms "
            f"query_ms={t_query_ms:.2f}ms total_ms={(time.perf_counter() - t_total_start)*1000:.1f}ms hits=0"
        )
        return [], 0.0

    max_score = max([hit.score for hit in points])
    docs = [
        {
            "id": f"chunk_{hit.id}",
            "text": hit.payload["text"],
            "score": hit.score,
            "source_file": hit.payload.get("source_file", hit.payload.get("source", "Knowledge Base")),
            "chunk_index": hit.payload.get("chunk_index", 0),
        }
        for hit in points
        if hit.score >= min_score_threshold and hit.payload and "text" in hit.payload
    ]
    logger.info(
        f"[PERF][qdrant_local] init_ms={t_init_ms:.2f}ms embed_ms={t_embed_ms:.1f}ms "
        f"query_ms={t_query_ms:.2f}ms total_ms={(time.perf_counter() - t_total_start)*1000:.1f}ms "
        f"hits={len(docs)} max_score={max_score:.3f}"
    )

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
                payload=payload,
            )
        )

    await q_client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )
    return len(points)
