import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import boto3
from openai import OpenAI
from pinecone import Pinecone  # SDK docs show pc.Index(host=...) then upsert

s3 = boto3.client("s3")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def s3_parse_uri(s3_uri: str) -> Tuple[str, str]:
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"S3 URI must start with s3://, got: {s3_uri}")
    rest = s3_uri[5:]
    bucket, key = rest.split("/", 1)
    return bucket, key


def s3_read_json(bucket: str, key: str) -> Dict[str, Any]:
    obj = s3.get_object(Bucket=bucket, Key=key)
    return json.loads(obj["Body"].read().decode("utf-8"))


def s3_put_json(bucket: str, key: str, data: Dict[str, Any]) -> None:
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(data, ensure_ascii=False).encode("utf-8"),
        ContentType="application/json",
    )


def safe_str(x: Any) -> str:
    # Pinecone metadata cannot contain null; keep strings always.
    return "" if x is None else str(x)


def build_vector_id(doc_id: str, page: Any, img_idx: Any) -> str:
    # Stable ID for easy debugging in retrieval / evidence display
    # Example: img::abc123::p0003::i02
    p = int(page) if page is not None else -1
    i = int(img_idx) if img_idx is not None else -1
    return f"img::{doc_id}::p{p:04d}::i{i:02d}"


def chunk_list(items: List[Any], n: int) -> List[List[Any]]:
    return [items[i : i + n] for i in range(0, len(items), n)]


def lambda_handler(event, context):
    """
    Input event example:
    {
      "doc_id": "abc123",
      "run_id": "run789",
      "image_captions_s3_uri": "s3://sec-rag-ai-system/derived/abc123/image_captions.json",
      "output_bucket": "sec-rag-ai-system",
      "output_prefix": "derived",

      "openai_embed_model": "text-embedding-3-small",
      "pinecone_index_host": "YOUR_INDEX_HOST",
      "pinecone_namespace": "sec"
    }
    """

    # ---- Required inputs ----
    doc_id = event["doc_id"]
    run_id = event.get("run_id", "")

    image_captions_s3_uri = event["image_captions_s3_uri"]
    output_bucket = event.get("output_bucket", "sec-rag-ai-system")
    output_prefix = str(event.get("output_prefix", "derived")).rstrip("/")

    # ---- Config from event OR env ----
    openai_model = event.get("openai_embed_model") or os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")

    pinecone_host = event.get("pinecone_index_host") or os.environ.get("PINECONE_INDEX_HOST")
    pinecone_namespace = event.get("pinecone_namespace") or os.environ.get("PINECONE_NAMESPACE", "sec")

    if not pinecone_host:
        raise ValueError("Missing pinecone_index_host (event) or PINECONE_INDEX_HOST (env).")

    openai_key = os.environ.get("OPENAI_API_KEY")
    pinecone_key = os.environ.get("PINECONE_API_KEY")

    if not openai_key:
        raise ValueError("Missing OPENAI_API_KEY env var.")
    if not pinecone_key:
        raise ValueError("Missing PINECONE_API_KEY env var.")

    # ---- Load captions JSON ----
    in_bucket, in_key = s3_parse_uri(image_captions_s3_uri)
    payload = s3_read_json(in_bucket, in_key)
    captions = payload.get("captions", [])

    # Filter non-empty captions (OpenAI embedding input cannot be empty).
    items = []
    for c in captions:
        caption_text = (c.get("caption") or "").strip()
        if not caption_text:
            continue
        items.append({
            "image_s3_uri": c.get("image_s3_uri"),
            "page": c.get("page"),
            "img_idx": c.get("img_idx"),
            "caption": caption_text.replace("\n", " ")
        })

    # ---- Init clients ----
    oai = OpenAI(api_key=openai_key)
    pc = Pinecone(api_key=pinecone_key)
    idx = pc.Index(host=pinecone_host)  # Pinecone SDK pattern

    # ---- Embed in batches ----
    BATCH_SIZE = int(event.get("batch_size", 64))  # safe default
    embedded_ids = []
    upserted = 0

    for batch in chunk_list(items, BATCH_SIZE):
        texts = [x["caption"] for x in batch]

        # OpenAI embeddings: pass a list of strings
        resp = oai.embeddings.create(model=openai_model, input=texts)
        vectors = []

        for x, row in zip(batch, resp.data):
            vec_id = build_vector_id(doc_id, x.get("page"), x.get("img_idx"))

            # Pinecone metadata must be flat JSON + no null values
            metadata = {
                "doc_id": doc_id,
                "run_id": safe_str(run_id),
                "chunk_type": "image",
                "page": int(x["page"]) if x.get("page") is not None else -1,
                "img_idx": int(x["img_idx"]) if x.get("img_idx") is not None else -1,
                "image_s3_uri": safe_str(x.get("image_s3_uri")),
                "caption": x["caption"],  # keep for easy evidence display later
            }

            vectors.append((vec_id, row.embedding, metadata))
            embedded_ids.append(vec_id)

        # Upsert into Pinecone namespace
        idx.upsert(vectors=vectors, namespace=pinecone_namespace)
        upserted += len(vectors)

    # ---- Write manifest back to S3 ----
    created_at = utc_now_iso()
    manifest_key = f"{output_prefix}/{doc_id}/image_embedding_manifest.json"

    s3_put_json(output_bucket, manifest_key, {
        "doc_id": doc_id,
        "run_id": run_id,
        "created_at": created_at,
        "openai_embed_model": openai_model,
        "pinecone_namespace": pinecone_namespace,
        "pinecone_index_host": pinecone_host,
        "num_captions_total": len(captions),
        "num_captions_embedded": len(items),
        "num_upserted": upserted,
        "vector_ids": embedded_ids[:2000],  # keep file reasonable; adjust if you want
    })

    return {
        "ok": True,
        "doc_id": doc_id,
        "run_id": run_id,
        "num_captions_embedded": len(items),
        "num_upserted": upserted,
        "image_embedding_manifest_s3_uri": f"s3://{output_bucket}/{manifest_key}",
    }