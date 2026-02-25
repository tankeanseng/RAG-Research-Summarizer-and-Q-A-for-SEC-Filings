from __future__ import annotations
from typing import Dict, Any, Optional
import re


_SUFFIX_RE = re.compile(r"^(?P<prefix>.+)-(?P<suffix>\d{5})$")


def extract_chunk_id(ch: Dict[str, Any]) -> Optional[str]:
    # Your chunks.json uses "chunk_id"
    v = ch.get("chunk_id")
    if isinstance(v, str) and v.strip():
        return v.strip()
    return None


def extract_chunk_text(ch: Dict[str, Any]) -> str:
    # Your chunks.json uses "content" (not "text")
    v = ch.get("content") or ch.get("text") or ch.get("chunk_text") or ""
    return v.strip() if isinstance(v, str) else ""


def build_chunk_lookup(chunks_json: Any) -> Dict[str, Dict[str, Any]]:
    if isinstance(chunks_json, dict) and "chunks" in chunks_json:
        chunks = chunks_json["chunks"]
    elif isinstance(chunks_json, list):
        chunks = chunks_json
    else:
        raise ValueError("Unexpected chunks.json format. Expect list or {chunks:[...]}")

    lookup: Dict[str, Dict[str, Any]] = {}
    for c in chunks:
        if not isinstance(c, dict):
            continue
        cid = extract_chunk_id(c)
        if cid:
            lookup[cid] = c
    return lookup


def canonicalize_candidate_id(candidate_id: str, doc_id: str) -> str:
    """
    Your clean_candidate_chunk_ids can have different UUID prefixes but same 5-digit suffix:
      1816d...-00025  ->  <doc_id>-00025
    This makes them match the chunk_id keys in chunks.json (doc-local).
    """
    c = candidate_id.strip()
    if c.startswith(doc_id + "-"):
        return c
    m = _SUFFIX_RE.match(c)
    if m:
        return f"{doc_id}-{m.group('suffix')}"
    return c


def dedup_preserve_order(ids: list[str]) -> list[str]:
    seen = set()
    out = []
    for x in ids:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out