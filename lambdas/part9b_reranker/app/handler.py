from __future__ import annotations

import os
import re
import time
from typing import Dict, Any, List

from app.s3io import s3_get_json, s3_put_json, s3_list_keys
from app.util import (
    build_chunk_lookup,
    extract_chunk_text,
    canonicalize_candidate_id,
    dedup_preserve_order,
)
from app.rerank import Reranker

_QFILE_RE = re.compile(r"/retrieval/queries/(q_\d+)\.json$")


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name, str(default)).strip()
    try:
        return int(v)
    except Exception:
        return default


def _norm_doc_id(event: Dict[str, Any]) -> str:
    doc_id = event.get("doc_id") or event.get("document_id") or ""
    if not doc_id or not isinstance(doc_id, str):
        raise ValueError("Missing event.doc_id (string)")
    return doc_id.strip()


def _load_questions_from_queries_folder(bucket: str, doc_id: str) -> List[Dict[str, Any]]:
    prefix = f"derived/{doc_id}/retrieval/queries/"
    keys = s3_list_keys(bucket, prefix)

    q_keys = [k for k in keys if _QFILE_RE.search(k)]
    if not q_keys:
        raise ValueError(f"No query files found under s3://{bucket}/{prefix}")

    def _qnum(key: str) -> int:
        m = _QFILE_RE.search(key)
        if not m:
            return 10**9
        q_id = m.group(1)  # q_01
        try:
            return int(q_id.split("_")[1])
        except Exception:
            return 10**9

    q_keys.sort(key=_qnum)

    out = []
    for k in q_keys:
        data = s3_get_json(bucket, k)
        q_id = data.get("q_id")
        base_q = data.get("base_question")
        if isinstance(q_id, str) and q_id.strip() and isinstance(base_q, str) and base_q.strip():
            out.append({"q_id": q_id.strip(), "question": base_q.strip(), "query_file_key": k})

    if not out:
        raise ValueError(f"Query files exist but no valid (q_id, base_question) parsed under {prefix}")
    return out


def _load_clean_candidate_ids(bucket: str, doc_id: str, q_id: str) -> List[str]:
    key = f"derived/{doc_id}/security/doc_injection/clean_candidates/{q_id}.json"
    data = s3_get_json(bucket, key)

    # Your real schema:
    # { ..., "clean_candidate_chunk_ids": [ "....-00025", ... ] }
    ids = data.get("clean_candidate_chunk_ids") if isinstance(data, dict) else None
    if not isinstance(ids, list):
        raise ValueError(f"Unexpected clean_candidates format for q_id={q_id}")

    out = []
    for cid in ids:
        if isinstance(cid, str) and cid.strip():
            out.append(cid.strip())
    return out


def lambda_handler(event: Dict[str, Any], context):
    t0 = time.time()

    bucket = os.environ.get("BUCKET", "").strip()
    if not bucket:
        raise ValueError("Missing env BUCKET")

    doc_id = _norm_doc_id(event)
    top_n = int(event.get("top_n") or _env_int("TOP_N", 12))
    batch_size = int(event.get("batch_size") or _env_int("BATCH_SIZE", 16))

    # 1) Questions
    questions = _load_questions_from_queries_folder(bucket, doc_id)

    # 2) Chunks
    chunks_key = f"derived/{doc_id}/chunks.json"
    chunks_json = s3_get_json(bucket, chunks_key)
    chunks_lookup = build_chunk_lookup(chunks_json)

    # Build chunk_texts from "content"
    chunk_texts = {}
    for cid, ch in chunks_lookup.items():
        txt = extract_chunk_text(ch)
        if txt:
            chunk_texts[cid] = txt

    reranker = Reranker()

    summary = {
        "doc_id": doc_id,
        "bucket": bucket,
        "chunks_key": chunks_key,
        "model_source": reranker.model_source,
        "top_n": top_n,
        "batch_size": batch_size,
        "questions_total": len(questions),
        "questions_processed": 0,
        "questions_failed": 0,
        "elapsed_sec": None,
        "outputs": [],
    }

    for q in questions:
        q_id = q["q_id"]
        question = q["question"]
        try:
            raw_ids = _load_clean_candidate_ids(bucket, doc_id, q_id)

            # Canonicalize to doc-local chunk_id and dedup
            canon_ids = [canonicalize_candidate_id(x, doc_id) for x in raw_ids]
            canon_ids = dedup_preserve_order(canon_ids)

            # Convert to candidate dicts expected by reranker
            candidates = [{"chunk_id": cid} for cid in canon_ids]

            reranked, dbg = reranker.rerank(
                query=question,
                candidates=candidates,
                chunk_texts=chunk_texts,
                top_n=top_n,
                batch_size=batch_size,
            )

            out_key = f"derived/{doc_id}/rerank/{q_id}.json"
            payload = {
                "doc_id": doc_id,
                "q_id": q_id,
                "question": question,
                "source_query_file": q.get("query_file_key"),
                "reranker": {
                    "model_source": reranker.model_source,
                    "top_n": top_n,
                    "batch_size": batch_size,
                },
                "debug": {
                    **dbg,
                    "raw_candidate_count": len(raw_ids),
                    "canon_candidate_count": len(canon_ids),
                },
                "reranked": reranked,
            }
            s3_put_json(bucket, out_key, payload)

            summary["questions_processed"] += 1
            summary["outputs"].append({"q_id": q_id, "s3_key": out_key, "returned": dbg.get("returned", 0)})

        except Exception as e:
            summary["questions_failed"] += 1
            summary["outputs"].append({"q_id": q_id, "error": str(e)})

    summary["elapsed_sec"] = round(time.time() - t0, 3)

    # Optional: write a summary file
    s3_put_json(bucket, f"derived/{doc_id}/rerank/_summary.json", summary)
    return summary