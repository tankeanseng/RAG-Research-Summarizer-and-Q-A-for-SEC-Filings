import json
import os
from datetime import datetime, timezone

import boto3
import fitz  # PyMuPDF
import pdfplumber

s3 = boto3.client("s3")

def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()

def s3_parse_uri(s3_uri: str):
    # s3://bucket/key
    if not s3_uri.startswith("s3://"):
        raise ValueError("S3 URI must start with s3://")
    rest = s3_uri[5:]
    bucket, key = rest.split("/", 1)
    return bucket, key

def looks_like_heading(text: str) -> bool:
    t = " ".join((text or "").split()).strip()
    if len(t) < 6:
        return False
    if len(t) <= 80 and (t.isupper() or t.endswith(":")):
        return True
    # SEC-ish patterns: "Item 1", "ITEM 1A"
    t_lower = t.lower()
    if t_lower.startswith("item "):
        return True
    # "1. Something"
    if len(t) <= 120 and t[:3].strip().startswith(("1", "2", "3", "4", "5", "6", "7", "8", "9")) and "." in t[:6]:
        return True
    return False

def sort_key_bbox(elem):
    x0, y0, x1, y1 = elem["bbox"]
    return (y0, x0, y1, x1)

def table_to_pipe_text(rows):
    # Simple table->text conversion for dense embedding later
    if not rows:
        return ""
    max_cols = max((len(r) for r in rows if r), default=0)
    norm = [(r + [""] * (max_cols - len(r))) if r else [""] * max_cols for r in rows]
    lines = [" | ".join((c or "").strip() for c in row) for row in norm]
    return "\n".join(lines)

def flush_text_chunk(chunks, doc_id, run_id, buf, meta):
    if not buf:
        return
    content = "\n\n".join(buf).strip()
    if not content:
        return
    chunk_id = f"{doc_id}-{len(chunks)+1:05d}"
    chunks.append({
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "run_id": run_id,
        "chunk_type": "text",
        "page_start": meta["page_start"],
        "page_end": meta["page_end"],
        "section_hint": meta.get("section_hint"),
        "content": content,
    })

def put_json(bucket, key, obj):
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(obj, ensure_ascii=False).encode("utf-8"),
        ContentType="application/json",
    )

def lambda_handler(event, context):
    """
    Input event example:
    {
      "doc_id": "abcd-...",
      "run_id": "efgh-...",                 (optional; if missing we generate)
      "input_pdf_s3_uri": "s3://sec-rag-ai-system/raw/<doc_id>/input.pdf",
      "output_bucket": "sec-rag-ai-system",
      "output_prefix": "derived",
      "max_chars": 3000
    }
    """

    doc_id = event["doc_id"]
    run_id = event.get("run_id") or os.urandom(8).hex()
    input_pdf_s3_uri = event["input_pdf_s3_uri"]

    output_bucket = event.get("output_bucket", "sec-rag-ai-system")
    output_prefix = event.get("output_prefix", "derived").rstrip("/")
    max_chars = int(event.get("max_chars", 3000))

    in_bucket, in_key = s3_parse_uri(input_pdf_s3_uri)
    local_pdf = f"/tmp/{doc_id}.pdf"

    # Download PDF (Lambda /tmp is writable; size is limited but configurable)
    s3.download_file(in_bucket, in_key, local_pdf)

    created_at = utc_now_iso()

    # Open PDF
    doc = fitz.open(local_pdf)
    num_pages = doc.page_count

    chunks = []
    images_manifest = []

    # text buffer
    buf = []
    buf_chars = 0
    section_hint = None
    page_start = None
    page_end = None

    TEXT_PRESERVE_IMAGES = fitz.TEXT_PRESERVE_IMAGES  # image blocks only if this flag is set

    with pdfplumber.open(local_pdf) as pdf:
        for p in range(num_pages):
            page_num = p + 1
            page = doc.load_page(p)

            # --- 1) Extract PyMuPDF text blocks (bbox + text)
            # get_text("blocks") gives bbox + block type; use sort=True to get top-left-ish order
            raw_blocks = page.get_text("blocks", sort=True, flags=TEXT_PRESERVE_IMAGES)

            text_elems = []
            image_block_elems = []
            for b in raw_blocks:
                x0, y0, x1, y1, text, block_no, block_type = b
                bbox = (x0, y0, x1, y1)
                if block_type == 0:
                    t = (text or "").strip()
                    if t:
                        text_elems.append({
                            "type": "text",
                            "page": page_num,
                            "bbox": bbox,
                            "text": t,
                            "is_heading": looks_like_heading(t),
                        })
                elif block_type == 1:
                    # image blocks appear only with TEXT_PRESERVE_IMAGES
                    image_block_elems.append({
                        "type": "image_block",
                        "page": page_num,
                        "bbox": bbox,
                    })

            # --- 2) Extract tables with bbox via pdfplumber (find_tables + bbox + extract)
            # pdfplumber supports table extraction & debugging
            table_elems = []
            t_objs = pdf.pages[p].find_tables()
            for t_i, t in enumerate(t_objs, start=1):
                rows = t.extract()
                table_text = table_to_pipe_text(rows)
                x0, top, x1, bottom = t.bbox
                table_elems.append({
                    "type": "table",
                    "page": page_num,
                    "bbox": (x0, top, x1, bottom),
                    "table_index_on_page": t_i,
                    "table_text": table_text,
                    "preview_rows": rows[:5] if rows else [],
                })

            # --- 3) Extract actual embedded images as files (xref-based)
            # We'll upload them to S3 for later captioning (multimodal)
            saved_imgs = []
            for j, img in enumerate(page.get_images(full=True), start=1):
                xref = img[0]
                info = doc.extract_image(xref)
                if not info or "image" not in info:
                    continue
                ext = info.get("ext", "bin")
                img_bytes = info["image"]

                local_img = f"/tmp/{doc_id}_p{page_num:04d}_img{j:02d}.{ext}"
                with open(local_img, "wb") as f:
                    f.write(img_bytes)

                s3_key = f"{output_prefix}/{doc_id}/images/page{page_num:04d}_img{j:02d}.{ext}"
                s3.upload_file(local_img, output_bucket, s3_key)

                saved_imgs.append({
                    "page": page_num,
                    "img_idx": j,
                    "xref": xref,
                    "s3_uri": f"s3://{output_bucket}/{s3_key}"
                })

            images_manifest.extend(saved_imgs)

            # --- 4) Merge all elements into one ordered stream (top-to-bottom)
            elems = []
            elems.extend(text_elems)
            elems.extend(table_elems)

            # Image chunk creation:
            # We create an "image chunk" if we extracted any image files on this page OR there were image blocks.
            # For v1: each saved image becomes a chunk (clean mapping to caption later).
            for im in saved_imgs:
                elems.append({
                    "type": "image",
                    "page": page_num,
                    "bbox": (0, 0, 0, 0),  # bbox unknown for xref extraction; ok for now
                    "image_s3_uri": im["s3_uri"],
                    "img_idx": im["img_idx"],
                })

            elems.sort(key=sort_key_bbox)

            # --- 5) Chunking rules (structure-aware)
            for e in elems:
                if e["type"] == "text":
                    if page_start is None:
                        page_start = page_num
                    page_end = page_num

                    if e.get("is_heading"):
                        flush_text_chunk(chunks, doc_id, run_id, buf, {
                            "page_start": page_start,
                            "page_end": page_end,
                            "section_hint": section_hint
                        })
                        buf = []
                        buf_chars = 0
                        section_hint = e["text"][:120]  # store a short section hint

                    buf.append(e["text"])
                    buf_chars += len(e["text"])

                    if buf_chars >= max_chars:
                        flush_text_chunk(chunks, doc_id, run_id, buf, {
                            "page_start": page_start,
                            "page_end": page_end,
                            "section_hint": section_hint
                        })
                        buf = []
                        buf_chars = 0
                        page_start = None
                        page_end = None

                elif e["type"] == "table":
                    flush_text_chunk(chunks, doc_id, run_id, buf, {
                        "page_start": page_start or page_num,
                        "page_end": page_end or page_num,
                        "section_hint": section_hint
                    })
                    buf = []
                    buf_chars = 0
                    page_start = None
                    page_end = None

                    chunk_id = f"{doc_id}-{len(chunks)+1:05d}"
                    chunks.append({
                        "chunk_id": chunk_id,
                        "doc_id": doc_id,
                        "run_id": run_id,
                        "chunk_type": "table",
                        "page_start": page_num,
                        "page_end": page_num,
                        "section_hint": section_hint,
                        "content": e["table_text"],
                        "table_meta": {
                            "table_index_on_page": e["table_index_on_page"],
                            "bbox": e["bbox"],
                            "preview_rows": e["preview_rows"],
                        }
                    })

                elif e["type"] == "image":
                    flush_text_chunk(chunks, doc_id, run_id, buf, {
                        "page_start": page_start or page_num,
                        "page_end": page_end or page_num,
                        "section_hint": section_hint
                    })
                    buf = []
                    buf_chars = 0
                    page_start = None
                    page_end = None

                    chunk_id = f"{doc_id}-{len(chunks)+1:05d}"
                    chunks.append({
                        "chunk_id": chunk_id,
                        "doc_id": doc_id,
                        "run_id": run_id,
                        "chunk_type": "image",
                        "page_start": page_num,
                        "page_end": page_num,
                        "section_hint": section_hint,
                        "content": f"[IMAGE to caption later] {e['image_s3_uri']}",
                        "image_meta": {
                            "image_s3_uri": e["image_s3_uri"],
                            "img_idx": e["img_idx"]
                        }
                    })

    doc.close()

    # Flush remaining buffered text
    if buf:
        flush_text_chunk(chunks, doc_id, run_id, buf, {
            "page_start": page_start or 1,
            "page_end": page_end or num_pages,
            "section_hint": section_hint
        })

    # Write outputs to S3
    chunks_key = f"{output_prefix}/{doc_id}/chunks.json"
    images_key = f"{output_prefix}/{doc_id}/images_manifest.json"
    stats_key = f"{output_prefix}/{doc_id}/chunking_stats.json"

    put_json(output_bucket, chunks_key, {
        "doc_id": doc_id,
        "run_id": run_id,
        "created_at": created_at,
        "chunks": chunks
    })
    put_json(output_bucket, images_key, {
        "doc_id": doc_id,
        "run_id": run_id,
        "created_at": created_at,
        "images": images_manifest
    })
    put_json(output_bucket, stats_key, {
        "doc_id": doc_id,
        "run_id": run_id,
        "created_at": created_at,
        "num_pages": num_pages,
        "num_chunks": len(chunks),
        "chunk_type_counts": {
            "text": sum(1 for c in chunks if c["chunk_type"] == "text"),
            "table": sum(1 for c in chunks if c["chunk_type"] == "table"),
            "image": sum(1 for c in chunks if c["chunk_type"] == "image"),
        },
        "num_images_saved": len(images_manifest)
    })

    return {
        "ok": True,
        "doc_id": doc_id,
        "run_id": run_id,
        "chunks_s3_uri": f"s3://{output_bucket}/{chunks_key}",
        "images_manifest_s3_uri": f"s3://{output_bucket}/{images_key}",
        "chunking_stats_s3_uri": f"s3://{output_bucket}/{stats_key}",
        "num_chunks": len(chunks),
        "num_images_saved": len(images_manifest)
    }