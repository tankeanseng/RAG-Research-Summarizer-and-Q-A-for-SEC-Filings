import json
from datetime import datetime, timezone

import boto3
import fitz  # PyMuPDF
import pdfplumber

s3 = boto3.client("s3")

def _utc_now_iso():
    return datetime.now(timezone.utc).isoformat()

def _s3_parse_uri(s3_uri: str):
    # s3://bucket/key
    if not s3_uri.startswith("s3://"):
        raise ValueError("input_s3_uri must start with s3://")
    path = s3_uri[5:]
    bucket, key = path.split("/", 1)
    return bucket, key

def lambda_handler(event, context):
    """
    Expected event:
    {
      "doc_id": "<doc_id>",
      "input_s3_uri": "s3://sec-rag-ai-system/raw/<doc_id>/input.pdf",
      "output_bucket": "sec-rag-ai-system",
      "output_prefix": "derived"
    }
    """

    doc_id = event["doc_id"]
    input_s3_uri = event["input_s3_uri"]
    output_bucket = event.get("output_bucket", "sec-rag-ai-system")
    output_prefix = event.get("output_prefix", "derived").rstrip("/")

    in_bucket, in_key = _s3_parse_uri(input_s3_uri)

    # Lambda gives you a writable temp folder at /tmp
    local_pdf_path = f"/tmp/{doc_id}.pdf"

    # 1) Download from S3
    s3.download_file(in_bucket, in_key, local_pdf_path)

    # 2) Text + image counts (PyMuPDF)
    doc = fitz.open(local_pdf_path)
    num_pages = doc.page_count

    pages_text = []
    images_by_page = []
    text_chars_total = 0
    num_images_total = 0

    for i in range(num_pages):
        page = doc.load_page(i)

        text = page.get_text("text") or ""
        pages_text.append({"page": i + 1, "text": text})
        text_chars_total += len(text)

        # Count images referenced by the page (PDF object refs, not ML “visual detection”)
        img_refs = page.get_images(full=False)
        img_count = len(img_refs)
        num_images_total += img_count
        images_by_page.append({"page": i + 1, "images": img_count})

    doc.close()

    # 3) Tables (pdfplumber)
    tables = []
    tables_by_page = []
    num_tables_total = 0

    with pdfplumber.open(local_pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            extracted = page.extract_tables() or []
            page_tables_count = 0

            for t_index, t in enumerate(extracted):
                if not t:
                    continue
                rows = len(t)
                cols = max((len(r) for r in t if r), default=0)
                if rows < 2 or cols < 2:
                    continue

                page_tables_count += 1
                num_tables_total += 1

                tables.append({
                    "page": i + 1,
                    "table_index_on_page": t_index + 1,
                    "rows": rows,
                    "cols": cols,
                    "preview": t[:5]  # small preview
                })

            tables_by_page.append({"page": i + 1, "tables": page_tables_count})

    created_at = _utc_now_iso()

    layout_stats = {
        "doc_id": doc_id,
        "input_s3_uri": input_s3_uri,
        "num_pages": num_pages,
        "num_images_total": num_images_total,
        "num_tables_total": num_tables_total,
        "text_chars_total": text_chars_total,
        "images_by_page": images_by_page,
        "tables_by_page": tables_by_page,
        "created_at": created_at
    }

    parsed_document = {
        "doc_id": doc_id,
        "input_s3_uri": input_s3_uri,
        "num_pages": num_pages,
        "pages": pages_text,
        "tables": tables,
        "created_at": created_at
    }

    stats_key = f"{output_prefix}/{doc_id}/layout_stats.json"
    parsed_key = f"{output_prefix}/{doc_id}/parsed_document.json"

    s3.put_object(
        Bucket=output_bucket,
        Key=stats_key,
        Body=json.dumps(layout_stats, ensure_ascii=False).encode("utf-8"),
        ContentType="application/json"
    )
    s3.put_object(
        Bucket=output_bucket,
        Key=parsed_key,
        Body=json.dumps(parsed_document, ensure_ascii=False).encode("utf-8"),
        ContentType="application/json"
    )

    return {
        "ok": True,
        "doc_id": doc_id,
        "layout_stats_s3_uri": f"s3://{output_bucket}/{stats_key}",
        "parsed_document_s3_uri": f"s3://{output_bucket}/{parsed_key}",
        "num_pages": num_pages,
        "num_images_total": num_images_total,
        "num_tables_total": num_tables_total
    }