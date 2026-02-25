#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$ROOT_DIR/dist"
mkdir -p "$OUT_DIR"
rm -f "$OUT_DIR/function.zip"

docker run --rm \
  --entrypoint /bin/bash \
  -v "$ROOT_DIR":/var/task \
  public.ecr.aws/lambda/python:3.11 \
  -lc "
    set -euo pipefail

    # install zip tool
    yum -y install zip >/dev/null

    cd /var/task
    rm -rf build && mkdir -p build

    # ensure pip exists
    python -m ensurepip --upgrade >/dev/null || true
    python -m pip install --upgrade pip >/dev/null

    # install deps into build/
    python -m pip install -r requirements.txt -t build >/dev/null

    # copy handler into build root (Lambda runtime expects /var/task/handler.py)
    cp app/handler.py build/handler.py

    cd build
    zip -r9 /var/task/dist/function.zip . >/dev/null

    echo \"Built: /var/task/dist/function.zip\"
  "