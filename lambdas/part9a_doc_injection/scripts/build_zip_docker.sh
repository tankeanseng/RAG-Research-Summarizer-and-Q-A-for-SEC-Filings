#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT_DIR/dist"
rm -f "$ROOT_DIR/dist/function.zip"

docker run --rm \
  --entrypoint /bin/bash \
  -v "$ROOT_DIR":/var/task \
  public.ecr.aws/lambda/python:3.11 \
  -lc "
    set -euo pipefail
    yum -y install zip >/dev/null
    cd /var/task
    rm -rf build && mkdir -p build
    python -m ensurepip --upgrade >/dev/null || true
    python -m pip install --upgrade pip >/dev/null
    python -m pip install -r requirements.txt -t build >/dev/null
    cp app/handler.py build/handler.py
    cd build
    zip -r9 /var/task/dist/function.zip . >/dev/null
    echo Built /var/task/dist/function.zip
  "