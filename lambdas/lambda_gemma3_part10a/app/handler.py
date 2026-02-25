import os
import json
import time
import re
import subprocess
import select
from typing import Any, Dict, List, Optional, Tuple

import boto3

s3 = boto3.client("s3")

# Stronger (but still short) system prompt
# You asked: "higher system prompt that tell it to answer based on context"
SYSTEM_PROMPT = (
    "You are a helpful assistant.\n"
    "Use the provided Context as your primary source.\n"
    "If the Context is insufficient, make a best-effort guess and keep it brief.\n"
)

CIT_RE = re.compile(r"\[([^\[\]]+)\]")


def utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def s3_get_json(bucket: str, key: str) -> Any:
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


def chunk_text_from_obj(ch: Dict[str, Any]) -> str:
    for k in ("content", "text", "chunk_text", "caption"):
        v = ch.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _read_vmrss_kb(pid: int) -> Optional[int]:
    try:
        with open(f"/proc/{pid}/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    return int(parts[1])
    except Exception:
        return None
    return None


def build_prompt(question: str, context_block: str) -> str:
    q = (question or "").strip()
    max_q = int(os.environ.get("MAX_QUESTION_CHARS", "600"))
    if len(q) > max_q:
        q = q[:max_q].rstrip()

    ctx = (context_block or "").strip()

    # Keep prompt compact and structured
    return (
        f"{SYSTEM_PROMPT}\n"
        f"Question:\n{q}\n\n"
        f"Context:\n{ctx}\n\n"
        f"Answer (brief):"
    )


def run_llama_streaming(prompt: str) -> Tuple[str, Dict[str, Any]]:
    """
    Streaming runner:
    - We do NOT enforce post-output safety caps until we have some output.
    - This keeps the “working” behavior but allows more context and longer answers.
    """
    llama_bin = os.environ.get("LLAMA_BIN", "/opt/llama/llama-cli")
    model_path = os.environ.get("MODEL_GGUF_PATH", "/opt/model/gemma3_merged_q4_k_m.gguf")

    # Slightly larger than your fast version, still safe-ish on Lambda CPU
    max_new_tokens = int(os.environ.get("MAX_NEW_TOKENS", "140"))
    ctx_size = int(os.environ.get("CTX_SIZE", "768"))
    threads = int(os.environ.get("THREADS", "4"))
    temp = float(os.environ.get("TEMPERATURE", "0.2"))
    top_p = float(os.environ.get("TOP_P", "0.9"))

    # Keep small to avoid OOM (batch-size trades memory for speed)
    n_batch = int(os.environ.get("N_BATCH", "8"))
    n_ubatch = int(os.environ.get("N_UBATCH", "4"))

    # “Don't stop until output exists”
    first_output_timeout_s = int(os.environ.get("FIRST_OUTPUT_TIMEOUT_S", "180"))
    min_output_chars = int(os.environ.get("MIN_OUTPUT_CHARS_BEFORE_GUARDS", "30"))

    # Safety caps only AFTER output exists
    post_output_time_budget_s = int(os.environ.get("POST_OUTPUT_TIME_BUDGET_S", "360"))
    rss_watermark_mb = int(os.environ.get("RSS_WATERMARK_MB", "9000"))
    rss_watermark_kb = rss_watermark_mb * 1024

    cmd = [
        llama_bin,
        "-m", model_path,
        "-p", prompt,
        "-n", str(max_new_tokens),
        "--ctx-size", str(ctx_size),
        "--threads", str(threads),
        "--top-p", str(top_p),
        "--temp", str(temp),
        "-b", str(n_batch),
        "-ub", str(n_ubatch),
        "--no-display-prompt",
        "--log-disable",
    ]

    t0 = time.time()
    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert p.stdout is not None
    assert p.stderr is not None

    out_parts: List[str] = []
    err_parts: List[str] = []

    got_output = False
    first_output_at = None
    killed_reason = None

    poller = select.poll()
    poller.register(p.stdout, select.POLLIN)
    poller.register(p.stderr, select.POLLIN)

    while True:
        rc = p.poll()
        now = time.time()

        # Drain available output
        events = poller.poll(200)
        for fd, ev in events:
            if ev & select.POLLIN:
                if fd == p.stdout.fileno():
                    s = p.stdout.readline()
                    if s:
                        out_parts.append(s)
                elif fd == p.stderr.fileno():
                    s = p.stderr.readline()
                    if s:
                        err_parts.append(s)

        current_out = "".join(out_parts).strip()
        if (not got_output) and (len(current_out) >= min_output_chars):
            got_output = True
            first_output_at = now

        # If finished, drain rest and exit
        if rc is not None:
            rest_out = p.stdout.read()
            rest_err = p.stderr.read()
            if rest_out:
                out_parts.append(rest_out)
            if rest_err:
                err_parts.append(rest_err)
            break

        # Before output exists: wait longer, do not kill for time/mem
        if not got_output:
            if (now - t0) > first_output_timeout_s:
                killed_reason = f"no_output_after_{first_output_timeout_s}s"
                p.kill()
                break
            continue

        # After output exists: enforce post-output time budget
        if first_output_at is not None and (now - first_output_at) > post_output_time_budget_s:
            killed_reason = f"post_output_time_budget_{post_output_time_budget_s}s"
            p.kill()
            break

        # After output exists: enforce RSS watermark
        rss_kb = _read_vmrss_kb(p.pid)
        if rss_kb is not None and rss_kb > rss_watermark_kb:
            killed_reason = f"rss_watermark_{rss_watermark_mb}mb"
            p.kill()
            break

    try:
        p.wait(timeout=2)
    except Exception:
        pass

    stdout = "".join(out_parts).strip()
    stderr_tail = ("".join(err_parts)).strip()[-1200:]

    # Defensive strip (should be unnecessary with --no-display-prompt)
    if stdout.startswith(prompt):
        stdout = stdout[len(prompt):].lstrip()

    dbg = {
        "cmd": cmd,
        "returncode": p.returncode,
        "elapsed_sec": round(time.time() - t0, 3),
        "stderr_tail": stderr_tail,
        "killed_reason": killed_reason,
        "got_output": got_output,
    }

    if not stdout:
        stdout = "Brief answer: unable to extract from context; providing a best-effort guess."

    return stdout, dbg


def discover_qids(bucket: str, derived_prefix: str, doc_id: str) -> List[str]:
    rerank_prefix = f"{derived_prefix}/{doc_id}/rerank/"
    keys = s3_list_keys(bucket, rerank_prefix)
    qids = []
    for k in keys:
        base = k.split("/")[-1]
        if base.endswith(".json") and base.startswith("q_") and base != "_summary.json":
            qids.append(base[:-5])
    return sorted(set(qids))


def load_rerank_question_and_first_chunk(bucket: str, derived_prefix: str, doc_id: str, q_id: str) -> Tuple[str, Optional[str]]:
    rerank_key = f"{derived_prefix}/{doc_id}/rerank/{q_id}.json"
    obj = s3_get_json(bucket, rerank_key)

    # Prefer retrieval/queries file if present, but keep minimal reads (fast)
    # Many of your pipelines store base_question in retrieval/queries.
    question = (obj.get("question") or obj.get("base_question") or obj.get("query") or "").strip()

    kept_top_n = obj.get("kept_top_n_chunk_ids")
    if isinstance(kept_top_n, list) and kept_top_n:
        return question, str(kept_top_n[0])

    reranked = obj.get("reranked") or []
    if isinstance(reranked, list) and reranked:
        first = reranked[0]
        if isinstance(first, dict) and first.get("chunk_id"):
            return question, str(first["chunk_id"])
        if isinstance(first, str):
            return question, first

    return question, None


def lambda_handler(event, context):
    t0 = time.time()

    bucket = event.get("bucket") or os.environ.get("BUCKET") or "sec-rag-ai-system"
    derived_prefix = (event.get("derived_prefix") or os.environ.get("DERIVED_PREFIX") or "derived").strip().strip("/")
    doc_id = event["doc_id"]
    run_id = event.get("run_id") or ""

    # Load chunks.json
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

    # Force ONLY 1 question even if multiple exist/passed
    q_ids = event.get("q_ids") or discover_qids(bucket, derived_prefix, doc_id)
    if not q_ids:
        summary = {
            "doc_id": doc_id,
            "run_id": run_id,
            "bucket": bucket,
            "derived_prefix": derived_prefix,
            "questions_total": 0,
            "questions_succeeded": 0,
            "questions_failed": 1,
            "elapsed_sec": round(time.time() - t0, 3),
            "outputs": [],
            "failed": [{"q_id": "", "error": "No q_ids found"}],
        }
        s3_put_json(bucket, f"{derived_prefix}/{doc_id}/answers/_summary.json", summary)
        return summary

    q_id = q_ids[0]

    outputs: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []

    try:
        question, first_chunk_id = load_rerank_question_and_first_chunk(bucket, derived_prefix, doc_id, q_id)

        # Only ONE chunk but allow more text than before
        context_text = ""
        included_chunk_ids: List[str] = []

        if first_chunk_id and first_chunk_id in chunks_by_id:
            ch = chunks_by_id[first_chunk_id]
            text = normalize_ws(chunk_text_from_obj(ch))

            # longer hard-trim (you requested this)
            max_chars = int(os.environ.get("MAX_CHARS_PER_CHUNK", "600"))
            text = text[:max_chars]

            # include minimal metadata if present (helps quality)
            p1 = ch.get("page_start")
            p2 = ch.get("page_end")
            page_info = ""
            if p1 is not None and p2 is not None:
                page_info = f"(pages {p1}-{p2}) "

            context_text = f"[{first_chunk_id}] {page_info}{text}"
            included_chunk_ids = [first_chunk_id]

        prompt = build_prompt(question=question, context_block=context_text)
        answer, dbg = run_llama_streaming(prompt)

        out_obj = {
            "meta": {
                "doc_id": doc_id,
                "run_id": run_id,
                "q_id": q_id,
                "bucket": bucket,
                "created_at": utc_now_iso(),
                "pipeline_version": os.environ.get("PIPELINE_VERSION", "v1.0.0"),
            },
            "question": question,
            "source": {
                "rerank_key": f"{derived_prefix}/{doc_id}/rerank/{q_id}.json",
                "chunks_key": f"{derived_prefix}/{doc_id}/chunks.json",
            },
            "generator": {
                "provider": "gemma3-270m",
                "runtime": "llama.cpp",
                "model_gguf_path": os.environ.get("MODEL_GGUF_PATH", "/opt/model/gemma3_merged_q4_k_m.gguf"),
                "prompt_style": "context_first_brief",
            },
            "context": {
                "max_context_chunks": 1,
                "max_chars_per_chunk": int(os.environ.get("MAX_CHARS_PER_CHUNK", "600")),
                "included_chunk_ids": included_chunk_ids,
            },
            "answer": answer,
            "citations": [],  # still not strict citations (faster / less brittle)
            "status": "ok",
            "debug": {
                "llama": dbg,
                "prompt_chars": len(prompt),
                "context_chars": len(context_text),
            },
        }

        out_key = f"{derived_prefix}/{doc_id}/answers/{q_id}.json"
        s3_put_json(bucket, out_key, out_obj)
        outputs.append({"q_id": q_id, "status": "ok", "out_key": out_key})

    except Exception as e:
        failed.append({"q_id": q_id, "error": str(e)})

    summary = {
        "doc_id": doc_id,
        "run_id": run_id,
        "bucket": bucket,
        "derived_prefix": derived_prefix,
        "questions_total": 1,
        "questions_succeeded": len(outputs),
        "questions_failed": len(failed),
        "elapsed_sec": round(time.time() - t0, 3),
        "outputs": outputs,
        "failed": failed,
    }
    s3_put_json(bucket, f"{derived_prefix}/{doc_id}/answers/_summary.json", summary)
    return summary