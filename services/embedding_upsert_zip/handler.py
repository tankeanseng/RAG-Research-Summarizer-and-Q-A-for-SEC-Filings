import json
import os
from datetime import datetime, timezone

import boto3
from openai import OpenAI
from pinecone.grpc import PineconeGRPC as Pinecone

s3 = boto3.client("s3")

def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()

def s3_parse_uri(s3_uri: str):
    if not s3_uri.startswith("s3://"):
        raise ValueError("S3 URI must start with s3://")
    rest = s3_uri[5:]
    bucket, key = rest.split("/", 1)
    return bucket, key

def s3_read_json(bucket: str, key: str):
    obj = s3.get_object(Bucket=bucket, Key=key)
    return json.loads(obj["Body"].read().decode("utf-8"))

def s3_put_json(bucket: str, key: str, data: dict):
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(data, ensure_ascii=False).encode("utf-8"),
        ContentType="application/json",
    )

def batch(iterable, size: int):
    buf = []
    for x in iterable:
        buf.append(x)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf

def lambda_handler(event, context):
    """
    event example:
    {
      "doc_id": "abc123",
      "run_id": "run789",
      "chunks_s3_uri": "s3://sec-rag-ai-system/derived/abc123/chunks.json",
      "output_bucket": "sec-rag-ai-system",
      "output_prefix": "derived",
      "embedding_model": "text-embedding-3-small",
      "pinecone_index": "sec-rag",
      "pinecone_namespace": "sec"
    }
    """

    doc_id = event["doc_id"]
    run_id = event.get("run_id")
    chunks_s3_uri = event["chunks_s3_uri"]

    output_bucket = event.get("output_bucket", "sec-rag-ai-system")
    output_prefix = event.get("output_prefix", "derived").rstrip("/")

    embedding_model = event.get("embedding_model", "text-embedding-3-small")
    pinecone_index_name = event.get("pinecone_index", "sec-rag")
    pinecone_namespace = event.get("pinecone_namespace", os.getenv("PINECONE_NAMESPACE", "sec"))

    # --- Secrets from env vars ---
    openai_key = os.environ["OPENAI_API_KEY"]
    pinecone_key = os.environ["PINECONE_API_KEY"]
    pinecone_host = os.environ["PINECONE_INDEX_HOST"]  # copy from Pinecone console

    client = OpenAI(api_key=openai_key)

    pc = Pinecone(api_key=pinecone_key)
    index = pc.Index(host=pinecone_host)

    # --- Load chunks ---
    in_bucket, in_key = s3_parse_uri(chunks_s3_uri)
    chunks_doc = s3_read_json(in_bucket, in_key)

    chunks = chunks_doc["chunks"]
    if not run_id:
        run_id = chunks_doc.get("run_id")

    created_at = utc_now_iso()

    # --- Prepare texts to embed ---
    # For now: embed only text/table chunks.
    # (image chunks will be embedded after you add vision captioning in a later part)
    to_embed = []
    skipped = []

    for c in chunks:
        ctype = c.get("chunk_type")
        if ctype in ("text", "table"):
            content = (c.get("content") or "").strip()
            if not content:
                skipped.append({"chunk_id": c.get("chunk_id"), "reason": "empty_content"})
                continue
            to_embed.append(c)
        else:
            skipped.append({"chunk_id": c.get("chunk_id"), "reason": f"skip_type_{ctype}"})

    # --- Embed + upsert in batches ---
    # Batch size: keep modest for rate limits and payload size.
    EMBED_BATCH = 64
    UPSERT_BATCH = 64

    embedded_count = 0
    vectors_to_upsert = []

    for batch_chunks in batch(to_embed, EMBED_BATCH):
        inputs = [c["content"] for c in batch_chunks]

        # OpenAI embeddings API (vector length default is 1536 for text-embedding-3-small) - verify index dim matches.
        emb = client.embeddings.create(
            model=embedding_model,
            input=inputs,
        )

        for c, e in zip(batch_chunks, emb.data):
            vec = e.embedding
            chunk_id = c["chunk_id"]

            # Store metadata for filtering and evidence display
            meta = {
                "doc_id": doc_id,
                "run_id": run_id,
                "chunk_id": chunk_id,
                "chunk_type": c.get("chunk_type"),
                "page_start": c.get("page_start"),
                "page_end": c.get("page_end"),
                "section_hint": c.get("section_hint") or "",
            }

            vectors_to_upsert.append((chunk_id, vec, meta))
            embedded_count += 1

        # upsert periodically
        if len(vectors_to_upsert) >= UPSERT_BATCH:
            index.upsert(vectors=vectors_to_upsert, namespace=pinecone_namespace)
            vectors_to_upsert = []

    # flush remaining
    if vectors_to_upsert:
        index.upsert(vectors=vectors_to_upsert, namespace=pinecone_namespace)

    # --- Write manifest to S3 ---
    manifest_key = f"{output_prefix}/{doc_id}/embedding_manifest.json"
    manifest = {
        "doc_id": doc_id,
        "run_id": run_id,
        "created_at": created_at,
        "embedding_model": embedding_model,
        "pinecone_index": pinecone_index_name,
        "pinecone_namespace": pinecone_namespace,
        "embedded_chunks": embedded_count,
        "skipped_chunks": skipped,
        "source_chunks_s3_uri": chunks_s3_uri,
    }
    s3_put_json(output_bucket, manifest_key, manifest)

    return {
        "ok": True,
        "doc_id": doc_id,
        "run_id": run_id,
        "embedded_chunks": embedded_count,
        "skipped_chunks": len(skipped),
        "pinecone_namespace": pinecone_namespace,
        "embedding_manifest_s3_uri": f"s3://{output_bucket}/{manifest_key}",
    }