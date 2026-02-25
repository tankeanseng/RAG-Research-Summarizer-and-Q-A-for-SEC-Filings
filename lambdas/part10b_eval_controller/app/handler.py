import json
import os
import re
import time
from typing import Any, Dict, List, Tuple

import boto3
import requests

s3 = boto3.client("s3")

CIT_RE = re.compile(r"\[([^\[\]]+)\]")  # [id] or [id1, id2]


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


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def split_sentences(text: str) -> List[str]:
    t = normalize_ws(text)
    if not t:
        return []
    parts = re.split(r"(?<=[\.\?\!])\s+", t)
    return [p.strip() for p in parts if p.strip()]


def parse_citations(text: str) -> List[str]:
    ids: List[str] = []
    for m in CIT_RE.finditer(text or ""):
        inside = m.group(1)
        for piece in inside.split(","):
            cid = normalize_ws(piece)
            if cid:
                ids.append(cid)
    return ids


def chunk_text_from_obj(ch: Dict[str, Any]) -> str:
    for k in ("content", "text", "chunk_text", "caption"):
        v = ch.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def build_context_pack(
    doc_id: str,
    chunks_by_id: Dict[str, Dict[str, Any]],
    allowed_ids: List[str],
    max_chars_per_chunk: int,
) -> List[Dict[str, Any]]:
    ctx: List[Dict[str, Any]] = []
    for cid in allowed_ids:
        cid = str(cid)
        if not cid.startswith(f"{doc_id}-"):
            continue
        ch = chunks_by_id.get(cid)
        if not ch:
            continue
        txt = chunk_text_from_obj(ch)
        if not txt:
            continue
        ctx.append(
            {
                "chunk_id": cid,
                "page_start": ch.get("page_start"),
                "page_end": ch.get("page_end"),
                "content": txt[:max_chars_per_chunk],
            }
        )
    return ctx


def openai_post(payload: Dict[str, Any], timeout_s: int) -> Dict[str, Any]:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY env var")

    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com").rstrip("/")
    url = base + "/v1/responses"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

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
    txt = (resp.get("output_text") or "").strip()
    if txt:
        return txt
    parts: List[str] = []
    for item in resp.get("output", []) or []:
        for c in item.get("content", []) or []:
            if c.get("type") == "output_text":
                parts.append(c.get("text", ""))
    return ("\n".join(parts)).strip()


def deterministic_checks(
    doc_id: str,
    ans_obj: Dict[str, Any],
    chunks_by_id: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Relaxed: NO hard gate on citation_coverage.
    Hard gates are:
      - invalid_citations
      - doc_scope_violation
      - cited_chunk_missing_in_chunks_json
      - no_citations (new)
      - schema mismatches
    We still compute citation_coverage as an informational metric.
    """
    failed: List[str] = []
    metrics: Dict[str, Any] = {}

    required_paths = [
        ("meta", dict),
        ("question", str),
        ("answer", str),
        ("citations", list),
        ("status", str),
        ("context", dict),
        ("generator", dict),
        ("source", dict),
    ]
    for k, t in required_paths:
        if k not in ans_obj:
            failed.append(f"missing:{k}")
        else:
            if not isinstance(ans_obj[k], t):
                failed.append(f"type:{k}")

    meta = ans_obj.get("meta", {}) if isinstance(ans_obj.get("meta"), dict) else {}
    if meta.get("doc_id") != doc_id:
        failed.append("meta_doc_id_mismatch")

    ctx = ans_obj.get("context", {}) if isinstance(ans_obj.get("context"), dict) else {}
    included_ids = ctx.get("included_chunk_ids", [])
    if not isinstance(included_ids, list):
        included_ids = []
        failed.append("type:context.included_chunk_ids")

    allowed_set = set([str(x) for x in included_ids])

    answer_text = ans_obj.get("answer") or ""
    cited = parse_citations(answer_text)

    # --- NEW: require at least 1 citation (instead of coverage threshold)
    if ans_obj.get("status") == "ok" and len(cited) == 0:
        failed.append("no_citations")

    invalid = [c for c in cited if c not in allowed_set]
    doc_scope_violations = [c for c in cited if not str(c).startswith(f"{doc_id}-")]

    metrics["cited_count"] = len(cited)
    metrics["invalid_citations"] = invalid
    metrics["doc_scope_violations"] = doc_scope_violations

    if invalid:
        failed.append("invalid_citations")
    if doc_scope_violations:
        failed.append("doc_scope_violation")

    # informational coverage only (not a failure gate)
    sents = split_sentences(answer_text)
    with_cit = sum(1 for s in sents if CIT_RE.search(s))
    coverage = (with_cit / len(sents)) if sents else 0.0
    metrics["citation_coverage"] = round(coverage, 3)
    metrics["sentences_total"] = len(sents)

    missing_cited = [cid for cid in cited if str(cid) not in chunks_by_id]
    metrics["missing_cited_chunks"] = missing_cited[:50]
    if missing_cited:
        failed.append("cited_chunk_missing_in_chunks_json")

    ok = (len(failed) == 0)
    return {
        "pass": ok,
        "failed_rules": failed,
        "metrics": metrics,
        "allowed_chunk_ids_count": len(included_ids),
    }


def judge_eval_quality(
    question: str,
    answer: str,
    contexts: List[Dict[str, Any]],
    timeout_s: int,
) -> Dict[str, Any]:
    judge_model = os.environ.get("OPENAI_MODEL_JUDGE", "gpt-4o-mini-2024-07-18").strip()

    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "scores": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "faithfulness": {"type": "number"},
                    "answer_relevancy": {"type": "number"},
                    "context_precision": {"type": "number"},
                    "context_utilization": {"type": "number"},
                },
                "required": [
                    "faithfulness",
                    "answer_relevancy",
                    "context_precision",
                    "context_utilization",
                ],
            },
            "verdict": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "pass": {"type": "boolean"},
                    "reasons": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["pass", "reasons"],
            },
        },
        "required": ["scores", "verdict"],
    }

    ctx_lines: List[str] = []
    for c in contexts:
        cid = c.get("chunk_id")
        p1 = c.get("page_start")
        p2 = c.get("page_end")
        ctx_lines.append(f"[{cid}] (pages {p1}-{p2})")
        ctx_lines.append(c.get("content", ""))
        ctx_lines.append("")
    ctx_text = "\n".join(ctx_lines).strip()

    instructions = (
        "You are an evaluator for RAG answers over SEC filings.\n"
        "Score the answer using ONLY the provided contexts.\n"
        "Use 0.0 to 1.0 where 1.0 is best.\n"
        "Faithfulness: claims supported by contexts.\n"
        "Answer relevancy: directly answers the question.\n"
        "Context precision: contexts are relevant to answering.\n"
        "Context utilization: answer actually uses the contexts.\n"
        "Return ONLY valid JSON matching the schema.\n"
    )

    payload = {
        "model": judge_model,
        "instructions": instructions,
        "input": (
            f"Question:\n{question}\n\n"
            f"Answer:\n{answer}\n\n"
            f"Contexts:\n{ctx_text}\n"
        ),
        "text": {
            "format": {
                "type": "json_schema",
                "name": "sec_rag_quality_judge_v1",
                "strict": True,
                "schema": schema,
            }
        },
        "temperature": 0.0,
        "truncation": "disabled",
    }

    resp = openai_post(payload, timeout_s=timeout_s)
    raw = extract_output_text(resp)
    if not raw:
        raise RuntimeError("Judge returned empty output_text")

    parsed = json.loads(raw)
    parsed["_judge_model"] = judge_model
    parsed["_openai_response_id"] = resp.get("id")
    parsed["_usage"] = resp.get("usage", {})
    return parsed


def regenerate_answer(
    question: str,
    contexts: List[Dict[str, Any]],
    allowed_ids: List[str],
    attempt: int,
    timeout_s: int,
    fail_reasons: List[str],
) -> Tuple[str, Dict[str, Any]]:
    model = os.environ.get("OPENAI_MODEL_ANSWER", "gpt-5.2").strip()

    ctx_lines: List[str] = []
    for c in contexts:
        cid = c["chunk_id"]
        p1 = c.get("page_start")
        p2 = c.get("page_end")
        ctx_lines.append(f"[{cid}] (pages {p1}-{p2})")
        ctx_lines.append(c["content"])
        ctx_lines.append("")
    ctx_text = "\n".join(ctx_lines).strip()

    base_rules = (
        "You are an SEC filing Q&A assistant.\n"
        "You MUST answer using ONLY the provided context.\n"
        "Every factual sentence MUST end with citations in square brackets using chunk_ids, e.g. [<chunk_id>].\n"
        "You may cite multiple chunks like [id1, id2].\n"
        "You MUST NOT invent chunk_ids.\n"
        "Allowed chunk_ids are:\n"
        + ", ".join(allowed_ids) + "\n"
        "If the context is insufficient, say so and cite what evidence exists.\n"
    )

    extra = ""
    if attempt == 2:
        extra = (
            "IMPORTANT: Your previous output failed checks: "
            + ", ".join(fail_reasons[:10])
            + ". Fix them.\n"
            "Use short bullet points. Each bullet must end with citations.\n"
        )
    elif attempt >= 3:
        extra = (
            "CRITICAL: You have failed twice. Output MUST be citation-compliant.\n"
            "Format:\n"
            "1) Direct answer in 3-6 bullets (each bullet ends with citations)\n"
            "2) Evidence list (bullets, each with citations)\n"
            "Do not include anything without citations.\n"
        )

    payload = {
        "model": model,
        "instructions": base_rules + extra,
        "input": f"Question:\n{question}\n\nContext:\n{ctx_text}\n",
        "temperature": 0.2,
        "truncation": "disabled",
    }

    resp = openai_post(payload, timeout_s=timeout_s)
    answer = extract_output_text(resp)
    if not answer:
        raise RuntimeError("Regeneration produced empty answer")

    debug = {
        "model": model,
        "openai_response_id": resp.get("id"),
        "usage": resp.get("usage", {}),
        "attempt": attempt,
        "fail_reasons": fail_reasons,
    }
    return answer, debug


def user_facing_failure(reason_codes: List[str]) -> Tuple[str, str, List[str]]:
    rset = set(reason_codes)

    if "invalid_citations" in rset or "no_citations" in rset:
        return (
            "Citation verification failed",
            "We could not produce a citation-compliant answer after multiple attempts. Below is an evidence-only summary.",
            [
                "Try asking a narrower question so fewer claims are needed.",
                "Try again later; citation enforcement may improve after updates.",
            ],
        )

    if "judge_failed_thresholds" in rset:
        return (
            "Quality verification failed",
            "We could not produce a sufficiently grounded answer after multiple attempts. Below is an evidence-only summary.",
            ["Try asking a narrower question.", "Try again later."],
        )

    return (
        "System limitation",
        "We could not produce a fully verified answer for this question after multiple attempts.",
        ["Try again later.", "Try a narrower question."],
    )


def evidence_only_fallback(question: str, contexts: List[Dict[str, Any]], max_bullets: int = 6) -> str:
    bullets = []
    for c in contexts[:max_bullets]:
        snippet = normalize_ws(c.get("content", ""))[:240]
        cid = c["chunk_id"]
        if snippet:
            bullets.append(f"- {snippet} [{cid}]")
    if not bullets:
        return f"We don't have enough usable context to answer this question: {question}"
    return "Evidence-only snippets:\n" + "\n".join(bullets)


def evaluate_one_q(
    bucket: str,
    derived_prefix: str,
    doc_id: str,
    q_id: str,
    chunks_by_id: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:

    answer_key = f"{derived_prefix}/{doc_id}/answers/{q_id}.json"
    ans_obj = s3_get_json(bucket, answer_key)

    rerank_key = f"{derived_prefix}/{doc_id}/rerank/{q_id}.json"
    rerank_obj = s3_get_json(bucket, rerank_key)

    question = (ans_obj.get("question") or rerank_obj.get("question") or rerank_obj.get("base_question") or "").strip()

    ctx_obj = ans_obj.get("context", {}) if isinstance(ans_obj.get("context"), dict) else {}
    allowed_ids = ctx_obj.get("included_chunk_ids", [])
    if not isinstance(allowed_ids, list):
        allowed_ids = []

    max_chars_per_chunk = int(os.environ.get("MAX_CHARS_PER_CHUNK", "1400"))
    contexts = build_context_pack(doc_id, chunks_by_id, [str(x) for x in allowed_ids], max_chars_per_chunk=max_chars_per_chunk)
    allowed_ids_clean = [c["chunk_id"] for c in contexts]

    det = deterministic_checks(doc_id, ans_obj, chunks_by_id)

    attempts_max = int(os.environ.get("MAX_RETRIES", "3"))
    openai_timeout = int(os.environ.get("OPENAI_TIMEOUT_S", "60"))

    attempt_logs = []
    det_pass = det["pass"]

    attempt = 1
    while (not det_pass) and (attempt < attempts_max):
        attempt += 1

        regen_text, regen_debug = regenerate_answer(
            question=question,
            contexts=contexts,
            allowed_ids=allowed_ids_clean,
            attempt=attempt,
            timeout_s=openai_timeout,
            fail_reasons=det["failed_rules"],
        )

        attempt_key = f"{derived_prefix}/{doc_id}/answers_attempts/{q_id}/attempt_{attempt}.json"
        s3_put_json(
            bucket,
            attempt_key,
            {
                "doc_id": doc_id,
                "q_id": q_id,
                "attempt": attempt,
                "created_at": now_iso(),
                "regen_debug": regen_debug,
                "answer": regen_text,
            },
        )

        ans_obj["answer"] = regen_text
        ans_obj["status"] = "ok"
        ans_obj["citations"] = sorted(set([c for c in parse_citations(regen_text) if c in set(allowed_ids_clean)]))

        det = deterministic_checks(doc_id, ans_obj, chunks_by_id)
        det_pass = det["pass"]

        attempt_logs.append(
            {
                "attempt": attempt,
                "attempt_key": attempt_key,
                "det_pass": det_pass,
                "failed_rules": det["failed_rules"],
            }
        )

    final_answer = ans_obj.get("answer") or ""
    final_citations = ans_obj.get("citations") if isinstance(ans_obj.get("citations"), list) else []
    final_det = det

    run_judge = os.environ.get("RUN_JUDGE", "true").lower() == "true"
    judge = None
    judge_pass = False

    if run_judge and final_det["pass"]:
        judge = judge_eval_quality(question, final_answer, contexts, timeout_s=openai_timeout)

        thr_faith = float(os.environ.get("THRESH_FAITHFULNESS", "0.80"))
        thr_rel = float(os.environ.get("THRESH_ANSWER_RELEVANCY", "0.70"))
        thr_prec = float(os.environ.get("THRESH_CONTEXT_PRECISION", "0.60"))
        thr_util = float(os.environ.get("THRESH_CONTEXT_UTILIZATION", "0.60"))

        s = judge.get("scores", {})
        judge_pass = (
            float(s.get("faithfulness", 0.0)) >= thr_faith
            and float(s.get("answer_relevancy", 0.0)) >= thr_rel
            and float(s.get("context_precision", 0.0)) >= thr_prec
            and float(s.get("context_utilization", 0.0)) >= thr_util
        )
        judge["verdict"]["pass"] = bool(judge_pass)

    if (not final_det["pass"]) or (run_judge and final_det["pass"] and not judge_pass):
        reason_codes = list(final_det["failed_rules"])
        if run_judge and judge and not judge_pass:
            reason_codes.append("judge_failed_thresholds")

        category, user_reason, next_steps = user_facing_failure(reason_codes)
        fallback = evidence_only_fallback(question, contexts)

        final_payload = {
            "meta": {
                "doc_id": doc_id,
                "q_id": q_id,
                "created_at": now_iso(),
                "pipeline_version": os.environ.get("PIPELINE_VERSION", "v1.0.0"),
            },
            "status": "terminal_fail",
            "question": question,
            "answer": fallback,
            "citations": sorted(set(parse_citations(fallback))),
            "deterministic": final_det,
            "judge": judge,
            "attempts": {"max_retries": attempts_max, "attempt_logs": attempt_logs},
            "user_feedback": {"category": category, "reason": user_reason, "next_steps": next_steps},
            "source": {
                "answers_key": answer_key,
                "rerank_key": rerank_key,
                "chunks_key": f"{derived_prefix}/{doc_id}/chunks.json",
            },
        }

        quarantine_key = f"{derived_prefix}/{doc_id}/eval/quarantine/{q_id}.json"
        s3_put_json(bucket, quarantine_key, final_payload)

        return {"final": final_payload, "quarantine_key": quarantine_key, "final_status": "terminal_fail"}

    final_payload = {
        "meta": {
            "doc_id": doc_id,
            "q_id": q_id,
            "created_at": now_iso(),
            "pipeline_version": os.environ.get("PIPELINE_VERSION", "v1.0.0"),
        },
        "status": "ok",
        "question": question,
        "answer": final_answer,
        "citations": final_citations,
        "deterministic": final_det,
        "judge": judge,
        "attempts": {"max_retries": attempts_max, "attempt_logs": attempt_logs},
        "user_feedback": {
            "category": "Success",
            "reason": "Answer passed deterministic checks and judge quality evaluation.",
            "next_steps": [],
        },
        "source": {
            "answers_key": answer_key,
            "rerank_key": rerank_key,
            "chunks_key": f"{derived_prefix}/{doc_id}/chunks.json",
        },
    }
    return {"final": final_payload, "quarantine_key": None, "final_status": "ok"}


def lambda_handler(event, context):
    t0 = time.time()

    bucket = event.get("bucket") or os.environ.get("BUCKET") or "sec-rag-ai-system"
    derived_prefix = (event.get("derived_prefix") or os.environ.get("DERIVED_PREFIX") or "derived").strip().strip("/")
    doc_id = event["doc_id"]
    run_id = event.get("run_id") or ""

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

    q_ids = event.get("q_ids")
    if not q_ids:
        ans_prefix = f"{derived_prefix}/{doc_id}/answers/"
        keys = s3_list_keys(bucket, ans_prefix)
        q_ids = []
        for k in keys:
            if k.endswith(".json") and not k.endswith("_summary.json"):
                base = k.split("/")[-1]
                qid = base[:-5]
                if qid.startswith("q_"):
                    q_ids.append(qid)
        q_ids = sorted(q_ids)

    finals = []
    quarantined = []

    for q_id in q_ids:
        result = evaluate_one_q(bucket, derived_prefix, doc_id, q_id, chunks_by_id)
        final_payload = result["final"]

        final_key = f"{derived_prefix}/{doc_id}/final/{q_id}.json"
        s3_put_json(bucket, final_key, final_payload)

        finals.append({"q_id": q_id, "status": result["final_status"], "final_key": final_key})
        if result["quarantine_key"]:
            quarantined.append({"q_id": q_id, "quarantine_key": result["quarantine_key"]})

    overall_status = "ok"
    if any(x["status"] != "ok" for x in finals):
        overall_status = "partial" if any(x["status"] == "ok" for x in finals) else "terminal_fail"

    email_payload = {
        "doc_id": doc_id,
        "run_id": run_id,
        "bucket": bucket,
        "derived_prefix": derived_prefix,
        "overall_status": overall_status,
        "created_at": now_iso(),
        "results": [],
        "notes": [
            "This report includes deterministic checks and LLM-as-judge quality evaluation results.",
            "Citations refer to original document chunks.",
        ],
    }

    for item in finals:
        q_id = item["q_id"]
        final_obj = s3_get_json(bucket, item["final_key"])
        email_payload["results"].append(
            {
                "q_id": q_id,
                "status": final_obj.get("status"),
                "question": final_obj.get("question"),
                "answer": final_obj.get("answer"),
                "citations": final_obj.get("citations", []),
                "deterministic": final_obj.get("deterministic", {}),
                "judge": final_obj.get("judge"),
                "user_feedback": final_obj.get("user_feedback", {}),
                "final_key": item["final_key"],
            }
        )

    email_key = f"{derived_prefix}/{doc_id}/final/_email_payload.json"
    s3_put_json(bucket, email_key, email_payload)

    summary = {
        "doc_id": doc_id,
        "run_id": run_id,
        "bucket": bucket,
        "derived_prefix": derived_prefix,
        "q_total": len(q_ids),
        "overall_status": overall_status,
        "finals": finals,
        "quarantined": quarantined,
        "email_payload_key": email_key,
        "elapsed_sec": round(time.time() - t0, 3),
    }
    s3_put_json(bucket, f"{derived_prefix}/{doc_id}/eval/_summary.json", summary)
    return summary