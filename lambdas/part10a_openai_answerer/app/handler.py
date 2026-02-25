import json
import os
import re
import time
from typing import Any, Dict, List, Tuple, Optional

import boto3
import requests

s3 = boto3.client("s3")

CIT_RE = re.compile(r"\[([^\[\]]+)\]")  # matches [chunk_id] or [id1, id2]


# -----------------------------
# S3 helpers
# -----------------------------
def s3_get_json(bucket: str, key: str) -> Dict[str, Any]:
    obj = s3.get_object(Bucket=bucket, Key=key)
    return json.loads(obj["Body"].read().decode("utf-8"))


def s3_put_json(bucket: str, key: str, data: Any) -> None:
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8"),
        ContentType="application/json",
    )


def s3_list_keys(bucket: str, prefix: str) -> List[str]:
    keys: List[str] = []
    token = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix, "MaxKeys": 1000}
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        for it in resp.get("Contents", []) or []:
            keys.append(it["Key"])
        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
        else:
            break
    return keys


# -----------------------------
# text helpers
# -----------------------------
def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def chunk_text_from_obj(ch: Dict[str, Any]) -> str:
    """
    Your chunks.json may store text under different fields.
    We try common ones safely.
    """
    for k in ("content", "text", "chunk_text", "caption"):
        v = ch.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


# -----------------------------
# OpenAI (Responses API via raw HTTP)
# -----------------------------
def openai_post(path: str, payload: Dict[str, Any], timeout_s: int) -> Dict[str, Any]:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY env var")

    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com").rstrip("/")
    url = base + path

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Simple retry for transient network/5xx
    last_err = None
    for attempt in range(3):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
            if r.status_code >= 400:
                raise RuntimeError(f"OpenAI HTTP {r.status_code}: {r.text[:2000]}")
            return r.json()
        except Exception as e:
            last_err = e
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"OpenAI request failed after retries: {last_err}")


def extract_output_text(resp: Dict[str, Any]) -> str:
    # Responses API commonly provides output_text convenience field
    txt = (resp.get("output_text") or "").strip()
    if txt:
        return txt

    # fallback: traverse output blocks
    parts: List[str] = []
    for item in resp.get("output", []) or []:
        for c in item.get("content", []) or []:
            if c.get("type") == "output_text":
                parts.append(c.get("text", ""))
    return ("\n".join(parts)).strip()


# -----------------------------
# Citation validation & repair
# -----------------------------
def parse_citations(text: str) -> List[str]:
    ids: List[str] = []
    for m in CIT_RE.finditer(text or ""):
        inside = m.group(1)
        # allow comma-separated within bracket
        for piece in inside.split(","):
            cid = normalize_ws(piece)
            if cid:
                ids.append(cid)
    return ids


def validate_citations(answer: str, allowed_ids: set) -> Tuple[bool, List[str]]:
    cited = parse_citations(answer)
    invalid = [c for c in cited if c not in allowed_ids]
    return (len(invalid) == 0, invalid)


def build_context_pack(
    doc_id: str,
    chunks_by_id: Dict[str, Dict[str, Any]],
    reranked_chunk_ids: List[str],
    max_chunks: int,
    max_chars_per_chunk: int,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Returns:
      context_items: list of {chunk_id, page_start, page_end, content}
      kept_ids: chunk_ids actually included (exists + has text + doc_id prefix check)
    """
    context_items: List[Dict[str, Any]] = []
    kept_ids: List[str] = []

    for cid in reranked_chunk_ids:
        if len(context_items) >= max_chunks:
            break

        # defense-in-depth: ensure belongs to this doc
        if not cid.startswith(f"{doc_id}-"):
            continue

        ch = chunks_by_id.get(cid)
        if not ch:
            continue

        text = chunk_text_from_obj(ch)
        text = text[:max_chars_per_chunk].strip() if text else ""
        if not text:
            continue

        context_items.append({
            "chunk_id": cid,
            "page_start": ch.get("page_start"),
            "page_end": ch.get("page_end"),
            "content": text,
        })
        kept_ids.append(cid)

    return context_items, kept_ids


def call_openai_answer(
    question: str,
    context_items: List[Dict[str, Any]],
    allowed_ids: List[str],
    attempt_note: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Uses Responses API. Enforces citations by instruction + validator.
    """
    model = os.environ.get("OPENAI_MODEL_ANSWER", "gpt-5.2").strip()
    timeout_s = int(os.environ.get("OPENAI_TIMEOUT_S", "60"))

    # Build compact context string
    # Use a deterministic format so citations map cleanly.
    ctx_lines: List[str] = []
    for c in context_items:
        cid = c["chunk_id"]
        p1 = c.get("page_start")
        p2 = c.get("page_end")
        header = f"[{cid}] (pages {p1}-{p2})"
        ctx_lines.append(header)
        ctx_lines.append(c["content"])
        ctx_lines.append("")  # spacer
    ctx_text = "\n".join(ctx_lines).strip()

    # Strong, explicit instruction for citation format
    instructions = (
        "You are an SEC filing Q&A assistant.\n"
        "You MUST answer using ONLY the provided context.\n"
        "Every factual claim must end with citations in square brackets using chunk_ids, e.g. [<chunk_id>].\n"
        "You may cite multiple chunks like [id1, id2].\n"
        "You MUST NOT invent chunk_ids.\n"
        "Allowed chunk_ids are:\n"
        + ", ".join(allowed_ids) + "\n"
        "If the context is insufficient, say what is missing and answer conservatively with citations where possible.\n"
    )
    if attempt_note:
        instructions += "\nIMPORTANT: " + attempt_note + "\n"

    payload = {
        "model": model,
        "instructions": instructions,
        "input": (
            f"Question:\n{question}\n\n"
            f"Context:\n{ctx_text}\n\n"
            "Write a clear, structured answer. Use short paragraphs and bullet points when helpful."
        ),
        "temperature": 0.2,
        # safe default: do not auto-truncate silently
        "truncation": "disabled",
    }

    resp = openai_post("/v1/responses", payload, timeout_s=timeout_s)
    answer = extract_output_text(resp)
    if not answer:
        raise RuntimeError("OpenAI returned empty answer text")

    debug = {
        "model": model,
        "openai_response_id": resp.get("id"),
        "usage": resp.get("usage", {}),
    }
    return answer, debug


# -----------------------------
# Main per-question pipeline
# -----------------------------
def process_one_question(
    bucket: str,
    derived_prefix: str,
    doc_id: str,
    run_id: str,
    q_id: str,
    chunks_by_id: Dict[str, Dict[str, Any]],
    max_context_chunks: int,
    max_chars_per_chunk: int,
) -> Dict[str, Any]:

    rerank_key = f"{derived_prefix}/{doc_id}/rerank/{q_id}.json"
    rerank_obj = s3_get_json(bucket, rerank_key)

    question = (
        rerank_obj.get("question")
        or rerank_obj.get("base_question")
        or rerank_obj.get("query")
        or ""
    ).strip()

    # Accept multiple possible fields
    reranked_list = rerank_obj.get("reranked", []) or []
    kept_top_n = rerank_obj.get("kept_top_n_chunk_ids") or []

    reranked_chunk_ids: List[str] = []
    if isinstance(kept_top_n, list) and kept_top_n:
        reranked_chunk_ids = [str(x) for x in kept_top_n]
    else:
        for it in reranked_list:
            cid = it.get("chunk_id")
            if cid:
                reranked_chunk_ids.append(str(cid))

    # Build context pack
    context_items, kept_ids = build_context_pack(
        doc_id=doc_id,
        chunks_by_id=chunks_by_id,
        reranked_chunk_ids=reranked_chunk_ids,
        max_chunks=max_context_chunks,
        max_chars_per_chunk=max_chars_per_chunk,
    )

    allowed_set = set(kept_ids)

    # If nothing to answer with, still emit artifact
    if not context_items:
        return {
            "meta": {
                "doc_id": doc_id,
                "run_id": run_id,
                "q_id": q_id,
                "bucket": bucket,
                "rerank_key": rerank_key,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "pipeline_version": os.environ.get("PIPELINE_VERSION", "v1.0.0"),
            },
            "question": question,
            "answer": "",
            "citations": [],
            "status": "no_context",
            "debug": {
                "kept_context_chunks": 0,
                "scored_candidates_in_rerank": len(reranked_chunk_ids),
            },
        }

    # Call OpenAI once; if invalid citations, retry once with explicit repair note
    answer1, debug1 = call_openai_answer(
        question=question,
        context_items=context_items,
        allowed_ids=kept_ids,
        attempt_note=None,
    )
    ok, invalid = validate_citations(answer1, allowed_set)

    final_answer = answer1
    debug2 = None
    repaired = False

    if not ok:
        note = (
            "Your previous answer used INVALID citations: "
            + ", ".join(invalid[:20])
            + ". You MUST only use the allowed chunk_ids list."
        )
        answer2, debug2 = call_openai_answer(
            question=question,
            context_items=context_items,
            allowed_ids=kept_ids,
            attempt_note=note,
        )
        ok2, invalid2 = validate_citations(answer2, allowed_set)
        if ok2:
            final_answer = answer2
            repaired = True
            invalid = []
        else:
            # fail-safe: keep second answer but record invalids;
            # downstream judge (Part 10B later) can flag it.
            final_answer = answer2
            invalid = invalid2

    citations = sorted(set([c for c in parse_citations(final_answer) if c in allowed_set]))

    out = {
        "meta": {
            "doc_id": doc_id,
            "run_id": run_id,
            "q_id": q_id,
            "bucket": bucket,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "pipeline_version": os.environ.get("PIPELINE_VERSION", "v1.0.0"),
        },
        "question": question,
        "source": {
            "rerank_key": rerank_key,
            "chunks_key": f"{derived_prefix}/{doc_id}/chunks.json",
        },
        "generator": {
            "provider": "openai",
            "model": os.environ.get("OPENAI_MODEL_ANSWER", "gpt-5.2").strip(),
            "responses_api": True,
            "timeout_s": int(os.environ.get("OPENAI_TIMEOUT_S", "60")),
        },
        "context": {
            "max_context_chunks": max_context_chunks,
            "max_chars_per_chunk": max_chars_per_chunk,
            "included_chunk_ids": kept_ids,
        },
        "answer": final_answer,
        "citations": citations,
        "status": "ok" if (len(invalid) == 0) else "has_invalid_citations",
        "debug": {
            "kept_context_chunks": len(kept_ids),
            "rerank_candidates": len(reranked_chunk_ids),
            "repaired": repaired,
            "invalid_citations": invalid,
            "openai_call_1": debug1,
            "openai_call_2": debug2,
        },
    }
    return out


# -----------------------------
# Lambda entry
# -----------------------------
def lambda_handler(event, context):
    """
    event:
    {
      "doc_id": "...",
      "run_id": "...",
      "bucket": "sec-rag-ai-system",
      "derived_prefix": "derived",
      "q_ids": ["q_01","q_02"]   # optional; if absent, auto-detect from derived/<doc_id>/rerank/
    }
    """
    t0 = time.time()

    bucket = event.get("bucket") or os.environ.get("BUCKET") or "sec-rag-ai-system"
    derived_prefix = (event.get("derived_prefix") or os.environ.get("DERIVED_PREFIX") or "derived").strip().strip("/")

    doc_id = event["doc_id"]
    run_id = event.get("run_id") or ""

    max_context_chunks = int(os.environ.get("MAX_CONTEXT_CHUNKS", "12"))
    max_chars_per_chunk = int(os.environ.get("MAX_CHARS_PER_CHUNK", "1400"))

    # Load chunks.json and build lookup
    chunks_key = f"{derived_prefix}/{doc_id}/chunks.json"
    chunks_obj = s3_get_json(bucket, chunks_key)

    raw_chunks = chunks_obj.get("chunks") if isinstance(chunks_obj, dict) else None
    if not isinstance(raw_chunks, list):
        raise RuntimeError(f"chunks.json missing 'chunks' list at s3://{bucket}/{chunks_key}")

    chunks_by_id: Dict[str, Dict[str, Any]] = {}
    for ch in raw_chunks:
        cid = ch.get("chunk_id") or ch.get("id")
        if cid:
            chunks_by_id[str(cid)] = ch

    # Determine q_ids
    q_ids = event.get("q_ids")
    if not q_ids:
        rerank_prefix = f"{derived_prefix}/{doc_id}/rerank/"
        keys = s3_list_keys(bucket, rerank_prefix)
        q_ids = []
        for k in keys:
            if k.endswith(".json") and not k.endswith("_summary.json"):
                base = k.split("/")[-1]
                qid = base[:-5]  # strip .json
                if qid.startswith("q_"):
                    q_ids.append(qid)
        q_ids = sorted(q_ids)

    outputs: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []

    for q_id in q_ids:
        try:
            ans = process_one_question(
                bucket=bucket,
                derived_prefix=derived_prefix,
                doc_id=doc_id,
                run_id=run_id,
                q_id=q_id,
                chunks_by_id=chunks_by_id,
                max_context_chunks=max_context_chunks,
                max_chars_per_chunk=max_chars_per_chunk,
            )
            out_key = f"{derived_prefix}/{doc_id}/answers/{q_id}.json"
            s3_put_json(bucket, out_key, ans)
            outputs.append({
                "q_id": q_id,
                "status": ans.get("status"),
                "out_key": out_key,
                "citations": len(ans.get("citations") or []),
            })
        except Exception as e:
            failed.append({"q_id": q_id, "error": str(e)})

    summary = {
        "doc_id": doc_id,
        "run_id": run_id,
        "bucket": bucket,
        "derived_prefix": derived_prefix,
        "questions_total": len(q_ids),
        "questions_succeeded": len(outputs),
        "questions_failed": len(failed),
        "elapsed_sec": round(time.time() - t0, 3),
        "outputs": outputs,
        "failed": failed,
    }
    s3_put_json(bucket, f"{derived_prefix}/{doc_id}/answers/_summary.json", summary)

    return summary