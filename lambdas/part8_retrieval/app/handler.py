import os
import json
import time
import math
import re
from typing import Dict, List, Any

import boto3
import requests

s3 = boto3.client("s3")

# -----------------------------
# Utilities
# -----------------------------
def utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def s3_get_json(bucket: str, key: str) -> Dict[str, Any]:
    obj = s3.get_object(Bucket=bucket, Key=key)
    return json.loads(obj["Body"].read().decode("utf-8"))


def s3_put_json(bucket: str, key: str, payload: Dict[str, Any]) -> None:
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
        ContentType="application/json",
    )


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


# NEW: chunk text accessor (keeps compatibility with older schemas)
def get_chunk_text(ch: Dict[str, Any]) -> str:
    """
    Your chunks.json uses 'content' for chunk text. Fall back to other keys for safety.
    """
    for k in ("content", "text", "chunk_text", "caption"):
        v = ch.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return ""


# -----------------------------
# BM25 (pure python)
# -----------------------------
def build_bm25_index(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    tf: Dict[str, Dict[str, int]] = {}
    df: Dict[str, int] = {}
    doc_len: Dict[str, int] = {}

    for ch in chunks:
        cid = ch["chunk_id"]

        # CHANGED: use get_chunk_text() instead of ch.get("text","")
        toks = tokenize(normalize_ws(get_chunk_text(ch)))
        doc_len[cid] = len(toks)

        counts: Dict[str, int] = {}
        seen = set()
        for t in toks:
            counts[t] = counts.get(t, 0) + 1
            if t not in seen:
                df[t] = df.get(t, 0) + 1
                seen.add(t)
        tf[cid] = counts

    N = len(chunks)
    avgdl = (sum(doc_len.values()) / N) if N else 0.0
    return {"tf": tf, "df": df, "doc_len": doc_len, "avgdl": avgdl, "N": N}


def bm25_score(query: str, cid: str, index: Dict[str, Any], k1: float = 1.2, b: float = 0.75) -> float:
    tf = index["tf"].get(cid, {})
    df = index["df"]
    dl = index["doc_len"].get(cid, 0)
    avgdl = index["avgdl"] or 1.0
    N = index["N"] or 1

    score = 0.0
    for term in tokenize(query):
        if term not in tf:
            continue
        n_qi = df.get(term, 0)
        idf = math.log((N - n_qi + 0.5) / (n_qi + 0.5) + 1.0)
        f = tf[term]
        denom = f + k1 * (1 - b + b * (dl / avgdl))
        score += idf * (f * (k1 + 1)) / (denom if denom else 1.0)
    return score


def bm25_topk(query: str, chunks_by_id: Dict[str, Dict[str, Any]], index: Dict[str, Any], top_k: int) -> List[Dict[str, Any]]:
    scored = []
    for cid in chunks_by_id.keys():
        s = bm25_score(query, cid, index)
        if s > 0:
            scored.append((cid, s))
    scored.sort(key=lambda x: x[1], reverse=True)

    out = []
    for i, (cid, s) in enumerate(scored[:top_k], start=1):
        ch = chunks_by_id[cid]
        out.append({
            "rank": i,
            "bm25_score": float(round(s, 6)),
            "chunk_id": cid,
            "page_start": ch.get("page_start"),
            "section_path": ch.get("section_path", []),
        })
    return out


# -----------------------------
# OpenAI (no SDK): Responses + Embeddings
# -----------------------------
def openai_post(path: str, payload: Dict[str, Any], timeout_s: int = 45) -> Dict[str, Any]:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing env OPENAI_API_KEY")

    url = "https://api.openai.com" + path
    r = requests.post(
        url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=timeout_s,
    )
    if r.status_code >= 400:
        raise RuntimeError(f"OpenAI error {r.status_code}: {r.text}")
    return r.json()


def openai_generate_subqueries(base_question: str) -> List[Dict[str, str]]:
    """
    Uses Responses API. We request strict JSON in plain text to avoid schema/format pitfalls.
    """
    model = os.environ.get("OPENAI_MODEL_SUBQ", os.environ.get("OPENAI_MODEL", "gpt-5.2"))
    timeout_s = int(os.environ.get("OPENAI_TIMEOUT_S", "45"))

    instructions = (
        "Generate search sub-queries for a single-document SEC filing RAG system.\n"
        "Return ONLY valid JSON with schema:\n"
        "{\n"
        '  "sub_queries":[\n'
        '    {"type":"semantic","text":"..."},\n'
        '    {"type":"keyword","text":"..."},\n'
        '    {"type":"entity","text":"..."}\n'
        "  ]\n"
        "}\n"
        "Rules:\n"
        "- Exactly 3 sub_queries.\n"
        "- Each text <= 140 chars.\n"
        "- semantic: paraphrase broader meaning.\n"
        "- keyword: short bag-of-words style.\n"
        "- entity: include SEC section hints like Item 1A, MD&A where relevant.\n"
    )

    payload = {
        "model": model,
        "instructions": instructions,
        "input": f"Base question: {base_question}",
        "temperature": 0.2,
    }

    data = openai_post("/v1/responses", payload, timeout_s=timeout_s)

    text_out = data.get("output_text")
    if not text_out:
        parts = []
        for item in data.get("output", []):
            for c in item.get("content", []):
                if c.get("type") == "output_text":
                    parts.append(c.get("text", ""))
        text_out = "\n".join(parts).strip()

    if not text_out:
        raise RuntimeError("OpenAI returned no text for subquery generation")

    try:
        obj = json.loads(text_out)
        subs = obj.get("sub_queries", [])
        if not isinstance(subs, list) or len(subs) != 3:
            raise ValueError("sub_queries must be list length 3")
        cleaned = []
        for s in subs:
            cleaned.append({
                "type": normalize_ws(s.get("type", "")),
                "text": normalize_ws(s.get("text", "")),
            })
        return cleaned
    except Exception as e:
        raise RuntimeError(f"Failed to parse subquery JSON. Raw:\n{text_out}\nError: {e}")


def openai_embed_text(text: str) -> List[float]:
    """
    Uses Embeddings API with text-embedding-3-small by default.
    """
    text = normalize_ws(text)
    if not text:
        return []

    model = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    timeout_s = int(os.environ.get("OPENAI_TIMEOUT_S", "45"))

    dims_raw = os.environ.get("OPENAI_EMBED_DIMENSIONS", "").strip()
    payload: Dict[str, Any] = {"model": model, "input": text}
    if dims_raw:
        payload["dimensions"] = int(dims_raw)

    data = openai_post("/v1/embeddings", payload, timeout_s=timeout_s)
    emb = data["data"][0]["embedding"]
    return [float(x) for x in emb]


# -----------------------------
# Pinecone REST query
# -----------------------------
def pinecone_query(vector: List[float], top_k: int, doc_id: str) -> List[Dict[str, Any]]:
    host = os.environ.get("PINECONE_HOST", "").strip()
    api_key = os.environ.get("PINECONE_API_KEY", "").strip()
    namespace = os.environ.get("PINECONE_NAMESPACE", "sec").strip()

    if not host or not api_key:
        raise RuntimeError("Missing env PINECONE_HOST or PINECONE_API_KEY")

    url = host.rstrip("/") + "/query"

    payload = {
        "vector": vector,
        "topK": top_k,
        "includeMetadata": True,
        "namespace": namespace,
        "filter": {"doc_id": {"$eq": doc_id}},
    }

    r = requests.post(
        url,
        headers={"Api-Key": api_key, "Content-Type": "application/json"},
        json=payload,
        timeout=30,
    )
    if r.status_code >= 400:
        raise RuntimeError(f"Pinecone query error {r.status_code}: {r.text}")

    data = r.json()
    matches = data.get("matches", []) or []

    out = []
    for i, m in enumerate(matches, start=1):
        md = m.get("metadata", {}) or {}
        if (md.get("doc_id") or "").strip() != doc_id:
            continue
        out.append({
            "rank": i,
            "cosine": float(round(m.get("score", 0.0), 6)),
            "vector_id": m.get("id"),
            "chunk_id": md.get("chunk_id", ""),
            "page_start": md.get("page_start"),
            "section_path": md.get("section_path", []),
        })
    return out


# -----------------------------
# RRF fusion
# -----------------------------
def rrf_fuse(result_lists: List[List[Dict[str, Any]]], k: int = 60) -> List[Dict[str, Any]]:
    scores: Dict[str, float] = {}
    contrib: Dict[str, List[Dict[str, Any]]] = {}

    for lst in result_lists:
        for item in lst:
            cid = item["chunk_id"]
            rank = int(item["rank"])
            scores[cid] = scores.get(cid, 0.0) + (1.0 / (k + rank))
            contrib.setdefault(cid, []).append({"rank": rank})

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    out = []
    for r, (cid, s) in enumerate(fused, start=1):
        out.append({"rank": r, "rrf_score": float(round(s, 8)), "chunk_id": cid})
    return out


# -----------------------------
# Lambda handler
# -----------------------------
def lambda_handler(event, context):
    """
    event:
    {
      "doc_id": "...",
      "run_id": "...",
      "bucket": "sec-rag-ai-system",
      "derived_prefix": "derived"
    }
    """
    t0 = time.time()

    bucket = event.get("bucket") or os.environ.get("BUCKET") or "sec-rag-ai-system"
    derived_prefix = (event.get("derived_prefix") or os.environ.get("DERIVED_PREFIX") or "derived").strip().strip("/")

    doc_id = event["doc_id"]
    run_id = event["run_id"]

    pipeline_version = os.environ.get("PIPELINE_VERSION", "v1.0.0")
    top_k_each = int(os.environ.get("RETRIEVAL_TOP_K", "20"))
    rrf_k = int(os.environ.get("RRF_K", "60"))

    # Load effective questions
    questions_key = f"{derived_prefix}/{doc_id}/questions/effective_questions.json"
    qdoc = s3_get_json(bucket, questions_key)

    questions = None
    if isinstance(qdoc, dict):
        if "effective_questions" in qdoc:
            questions = qdoc["effective_questions"]
        elif "questions" in qdoc:
            questions = qdoc["questions"]
        else:
            questions = qdoc
    elif isinstance(qdoc, list):
        questions = qdoc
    elif isinstance(qdoc, str):
        try:
            parsed = json.loads(qdoc)
            qdoc = parsed
            if isinstance(qdoc, dict) and "effective_questions" in qdoc:
                questions = qdoc["effective_questions"]
            elif isinstance(qdoc, list):
                questions = qdoc
            else:
                questions = qdoc
        except Exception:
            raise RuntimeError(f"effective_questions.json is a string but not valid JSON. First 200 chars: {qdoc[:200]}")
    else:
        raise RuntimeError(f"Unexpected effective_questions.json type: {type(qdoc)}")

    if isinstance(questions, dict):
        questions = [questions]
    if not isinstance(questions, list):
        raise RuntimeError(f"Questions payload not a list after normalization: {type(questions)}")

    norm_questions = []
    for i, item in enumerate(questions):
        if isinstance(item, str):
            txt = normalize_ws(item)
            if not txt:
                continue
            norm_questions.append({"q_id": f"q_{i+1:02d}", "text": txt})
            continue
        if not isinstance(item, dict):
            continue

        qid = normalize_ws(str(item.get("q_id", "")))
        txt = normalize_ws(str(item.get("text", item.get("question", ""))))

        if not txt:
            continue
        if not qid:
            qid = f"q_{i+1:02d}"
        norm_questions.append({"q_id": qid, "text": txt})

    questions = norm_questions

    print(f"[Part8] Loaded questions: count={len(questions)}; sample={questions[:2]}")

    # Load chunks
    chunks_key = f"{derived_prefix}/{doc_id}/chunks.json"
    cdoc = s3_get_json(bucket, chunks_key)
    chunks = cdoc.get("chunks", [])
    chunks_by_id = {c["chunk_id"]: c for c in chunks}

    # BM25 stats
    bm25_idx = build_bm25_index(chunks)

    min_top_dense = float(os.environ.get("MIN_TOP_DENSE_COSINE", "0.35"))
    min_top_rrf = float(os.environ.get("MIN_TOP_RRF_SCORE", "0.01"))
    min_evidence = int(os.environ.get("MIN_EVIDENCE_COUNT", "3"))

    for q in questions:
        q_id = q["q_id"]
        base_q = normalize_ws(q.get("text", ""))

        subqs = openai_generate_subqueries(base_q)
        sub_queries = []
        for i, s in enumerate(subqs):
            sub_queries.append({
                "subq_id": f"{q_id}_{chr(ord('a') + i)}",
                "type": s["type"],
                "text": s["text"],
            })

        s3_put_json(bucket, f"{derived_prefix}/{doc_id}/retrieval/queries/{q_id}.json", {
            "meta": {"doc_id": doc_id, "run_id": run_id, "created_at": utc_now_iso(), "pipeline_version": pipeline_version},
            "q_id": q_id,
            "base_question": base_q,
            "sub_queries": sub_queries
        })

        all_bm25_lists = []
        all_dense_lists = []

        for sq in sub_queries:
            subq_id = sq["subq_id"]
            text = sq["text"]

            bm25_res = bm25_topk(text, chunks_by_id, bm25_idx, top_k_each)
            s3_put_json(bucket, f"{derived_prefix}/{doc_id}/retrieval/bm25/{q_id}/{subq_id}.json", {
                "meta": {"doc_id": doc_id, "run_id": run_id, "created_at": utc_now_iso(), "pipeline_version": pipeline_version},
                "q_id": q_id,
                "subq_id": subq_id,
                "top_k": top_k_each,
                "results": bm25_res
            })
            all_bm25_lists.append(bm25_res)

            vec = openai_embed_text(text)
            dense_res = pinecone_query(vec, top_k_each, doc_id=doc_id) if vec else []

            s3_put_json(bucket, f"{derived_prefix}/{doc_id}/retrieval/dense/{q_id}/{subq_id}.json", {
                "meta": {"doc_id": doc_id, "run_id": run_id, "created_at": utc_now_iso(), "pipeline_version": pipeline_version},
                "q_id": q_id,
                "subq_id": subq_id,
                "top_k": top_k_each,
                "results": dense_res
            })
            all_dense_lists.append(dense_res)

        fused = rrf_fuse(all_bm25_lists + all_dense_lists, k=rrf_k)

        contributions = {}
        for src_name, lists in [("bm25", all_bm25_lists), ("dense", all_dense_lists)]:
            for lst_i, lst in enumerate(lists):
                for item in lst:
                    cid = item["chunk_id"]
                    contributions.setdefault(cid, [])
                    contributions[cid].append({"source": src_name, "rank": item["rank"]})

        fused_out = []
        for item in fused[:top_k_each]:
            cid = item["chunk_id"]
            ch = chunks_by_id.get(cid, {})
            fused_out.append({
                "rank": item["rank"],
                "rrf_score": item["rrf_score"],
                "chunk_id": cid,
                "page_start": ch.get("page_start"),
                "section_path": ch.get("section_path", []),
                "contributions": contributions.get(cid, [])
            })

        s3_put_json(bucket, f"{derived_prefix}/{doc_id}/retrieval/fused/{q_id}.json", {
            "meta": {"doc_id": doc_id, "run_id": run_id, "created_at": utc_now_iso(), "pipeline_version": pipeline_version},
            "q_id": q_id,
            "rrf": {"k": rrf_k, "sources": ["bm25", "dense"], "sub_queries": len(sub_queries)},
            "fused_results": fused_out
        })

        top_dense = 0.0
        for lst in all_dense_lists:
            if lst:
                top_dense = max(top_dense, float(lst[0].get("cosine", 0.0)))

        top_rrf = float(fused_out[0]["rrf_score"]) if fused_out else 0.0
        evidence_count = len(fused_out)
        unique_sections = len(set([" > ".join(x.get("section_path", [])) for x in fused_out if x.get("section_path")]))

        decision = "answer"
        refusal_reason = None
        if (top_dense < min_top_dense) and (top_rrf < min_top_rrf or evidence_count < min_evidence):
            decision = "refuse"
            refusal_reason = "LOW_EVIDENCE"

        s3_put_json(bucket, f"{derived_prefix}/{doc_id}/retrieval/answerability/{q_id}.json", {
            "meta": {"doc_id": doc_id, "run_id": run_id, "created_at": utc_now_iso(), "pipeline_version": pipeline_version},
            "q_id": q_id,
            "decision": decision,
            "signals": {
                "top_dense_cosine": float(round(top_dense, 6)),
                "top_rrf_score": float(round(top_rrf, 8)),
                "evidence_count": evidence_count,
                "unique_sections": unique_sections
            },
            "thresholds": {
                "min_top_dense_cosine": min_top_dense,
                "min_top_rrf_score": min_top_rrf,
                "min_evidence_count": min_evidence
            },
            "refusal_reason": refusal_reason
        })

    elapsed_ms = int((time.time() - t0) * 1000)
    return {"ok": True, "doc_id": doc_id, "run_id": run_id, "questions_processed": len(questions), "elapsed_ms": elapsed_ms}