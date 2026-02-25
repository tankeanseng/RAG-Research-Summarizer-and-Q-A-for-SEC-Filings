"""
app.py — AWS Lambda (container image) BLIP captioner

What this does:
- Reads images_manifest.json from S3 (created by your Part 5 chunker)
- Downloads EVERY image listed there
- Runs BLIP image captioning
- Writes:
  - derived/<doc_id>/image_captions.json
  - derived/<doc_id>/image_caption_stats.json

Important Lambda filesystem rule:
- Container images must run on a read-only filesystem; only /tmp is writable.

Important Hugging Face cache rule:
- HF cache defaults to ~/.cache/... unless you set HF_HOME / HF_HUB_CACHE / TRANSFORMERS_CACHE / XDG_CACHE_HOME. 
"""

import json
import os
import shutil
from datetime import datetime, timezone
from io import BytesIO
from typing import Dict, Tuple, Any, List

import boto3
from PIL import Image

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration  # BLIP in Transformers docs


# ----------------------------
# 0) Hugging Face cache setup (Lambda-safe)
# ----------------------------
# Lambda container images: FS is read-only except /tmp. 
# So we force all HF caches to /tmp, and (optionally) copy a baked cache from /opt/hf into /tmp/hf.
RUNTIME_HF_HOME = "/tmp/hf"
BAKED_HF_HOME = "/opt/hf"  # where the Dockerfile pre-download step stores HF cache

os.environ["HF_HOME"] = RUNTIME_HF_HOME
os.environ["HF_HUB_CACHE"] = os.path.join(RUNTIME_HF_HOME, "hub")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(RUNTIME_HF_HOME, "transformers")
os.environ["XDG_CACHE_HOME"] = "/tmp"  # affects fallback cache resolution 

# If you baked the model into the image during docker build, copy it to /tmp on cold start
# so HF hub can create lock files / metadata under a writable path.
if os.path.exists(BAKED_HF_HOME) and not os.path.exists(RUNTIME_HF_HOME):
    # Copy once per cold start
    shutil.copytree(BAKED_HF_HOME, RUNTIME_HF_HOME)


# ----------------------------
# 1) Config
# ----------------------------
MODEL_ID = os.environ.get("BLIP_MODEL_ID", "Salesforce/blip-image-captioning-base")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Optional knobs (can override in the event)
DEFAULT_MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "40"))
DEFAULT_MAX_IMAGE_SIDE = int(os.environ.get("MAX_IMAGE_SIDE", "1024"))

# S3 client
s3 = boto3.client("s3")


# ----------------------------
# 2) Load model once (warm-start friendly)
# ----------------------------
# NOTE: use_safetensors=True avoids torch.load pickle path (and aligns with safer loading).
processor = BlipProcessor.from_pretrained(MODEL_ID, cache_dir=RUNTIME_HF_HOME)
model = BlipForConditionalGeneration.from_pretrained(
    MODEL_ID,
    use_safetensors=True,
    cache_dir=RUNTIME_HF_HOME,
).to(DEVICE)
model.eval()

# Optional: lightweight CPU optimization (safe to ignore if it fails)
if DEVICE == "cpu":
    try:
        model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    except Exception:
        pass


# ----------------------------
# 3) Helpers
# ----------------------------
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


def download_image_from_s3(s3_uri: str) -> Image.Image:
    b, k = s3_parse_uri(s3_uri)
    obj = s3.get_object(Bucket=b, Key=k)
    img_bytes = obj["Body"].read()
    return Image.open(BytesIO(img_bytes)).convert("RGB")


def resize_if_needed(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return img
    scale = max_side / float(m)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    return img.resize((new_w, new_h))


def caption_image(img: Image.Image, max_new_tokens: int) -> str:
    # BLIP captioning usage as documented in Transformers BLIP docs.
    inputs = processor(images=img, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    caption = processor.decode(out_ids[0], skip_special_tokens=True)
    return caption.strip()


# ----------------------------
# 4) Lambda handler
# ----------------------------
def lambda_handler(event, context):
    """
    Expected input event example:
    {
      "doc_id": "abc123",
      "run_id": "run789",
      "images_manifest_s3_uri": "s3://sec-rag-ai-system/derived/abc123/images_manifest.json",
      "output_bucket": "sec-rag-ai-system",
      "output_prefix": "derived",
      "max_new_tokens": 40,
      "max_image_side": 1024
    }
    """

    doc_id = event["doc_id"]
    run_id = event.get("run_id", "")

    images_manifest_s3_uri = event["images_manifest_s3_uri"]

    output_bucket = event.get("output_bucket", "sec-rag-ai-system")
    output_prefix = str(event.get("output_prefix", "derived")).rstrip("/")

    max_new_tokens = int(event.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS))
    max_image_side = int(event.get("max_image_side", DEFAULT_MAX_IMAGE_SIDE))

    in_bucket, in_key = s3_parse_uri(images_manifest_s3_uri)
    manifest = s3_read_json(in_bucket, in_key)

    images: List[Dict[str, Any]] = manifest.get("images", [])

    captions_out: List[Dict[str, Any]] = []
    num_errors = 0
    num_captioned = 0

    for im in images:
        image_s3_uri = im["s3_uri"]
        page = im.get("page")
        img_idx = im.get("img_idx")

        try:
            img = download_image_from_s3(image_s3_uri)
            img = resize_if_needed(img, max_image_side)

            cap = caption_image(img, max_new_tokens=max_new_tokens)
            if cap:
                num_captioned += 1

            captions_out.append(
                {
                    "image_s3_uri": image_s3_uri,
                    "page": page,
                    "img_idx": img_idx,
                    "caption": cap,
                }
            )
        except Exception as e:
            num_errors += 1
            captions_out.append(
                {
                    "image_s3_uri": image_s3_uri,
                    "page": page,
                    "img_idx": img_idx,
                    "caption": "",
                    "error": str(e)[:500],
                }
            )

    created_at = utc_now_iso()

    captions_key = f"{output_prefix}/{doc_id}/image_captions.json"
    stats_key = f"{output_prefix}/{doc_id}/image_caption_stats.json"

    s3_put_json(
        output_bucket,
        captions_key,
        {
            "doc_id": doc_id,
            "run_id": run_id,
            "created_at": created_at,
            "model_id": MODEL_ID,
            "device": DEVICE,
            "max_new_tokens": max_new_tokens,
            "max_image_side": max_image_side,
            "captions": captions_out,
        },
    )

    s3_put_json(
        output_bucket,
        stats_key,
        {
            "doc_id": doc_id,
            "run_id": run_id,
            "created_at": created_at,
            "num_images": len(images),
            "num_captioned": num_captioned,
            "num_errors": num_errors,
        },
    )

    return {
        "ok": True,
        "doc_id": doc_id,
        "run_id": run_id,
        "model_id": MODEL_ID,
        "num_images": len(images),
        "num_captioned": num_captioned,
        "num_errors": num_errors,
        "image_captions_s3_uri": f"s3://{output_bucket}/{captions_key}",
        "image_caption_stats_s3_uri": f"s3://{output_bucket}/{stats_key}",
    }