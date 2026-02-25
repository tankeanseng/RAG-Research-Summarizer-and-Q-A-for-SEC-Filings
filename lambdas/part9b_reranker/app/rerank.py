from __future__ import annotations
from typing import List, Dict, Any, Tuple
import os

# Writable cache dirs in Lambda
os.environ.setdefault("HF_HOME", "/tmp/hf")
os.environ.setdefault("HF_HUB_CACHE", "/tmp/hf/hub")
os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/hf/transformers")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class Reranker:
    def __init__(self):
        model_dir = os.environ.get("MODEL_DIR", "/opt/models/ms-marco-MiniLM-L6-v2")
        model_id = os.environ.get("MODEL_ID", "cross-encoder/ms-marco-MiniLM-L6-v2")

        self.cache_dir = os.environ.get("TRANSFORMERS_CACHE", "/tmp/hf/transformers")
        self.model_source = model_dir if os.path.exists(model_dir) else model_id

        local_only = os.path.exists(model_dir)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_source,
            use_fast=True,
            cache_dir=self.cache_dir,
            local_files_only=local_only,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_source,
            cache_dir=self.cache_dir,
            local_files_only=local_only,
        )

        self.device = torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        chunk_texts: Dict[str, str],
        top_n: int,
        batch_size: int = 16,
        max_length: int = 512,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        pairs = []
        keep = []

        for c in candidates:
            cid = c.get("chunk_id")
            if not cid:
                continue
            cid = str(cid).strip()
            txt = chunk_texts.get(cid, "")
            if not txt:
                continue
            keep.append(c)
            pairs.append((query, txt))

        if not pairs:
            return [], {"kept": 0, "scored": 0, "returned": 0, "model_source": self.model_source}

        scores: List[float] = []
        bs = max(1, int(batch_size))

        for i in range(0, len(pairs), bs):
            batch = pairs[i : i + bs]
            enc = self.tokenizer(
                [q for q, _ in batch],
                [p for _, p in batch],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            out = self.model(**enc)
            logits = out.logits.squeeze(-1).detach().cpu().tolist()
            if isinstance(logits, float):
                logits = [logits]
            scores.extend([float(x) for x in logits])

        scored = []
        for c, s in zip(keep, scores):
            cc = dict(c)
            cc["rerank_score"] = float(s)
            scored.append(cc)

        scored.sort(key=lambda x: x.get("rerank_score", -1e9), reverse=True)
        reranked = scored[: max(0, int(top_n))]

        dbg = {
            "kept": len(keep),
            "scored": len(scored),
            "returned": len(reranked),
            "model_source": self.model_source,
        }
        return reranked, dbg