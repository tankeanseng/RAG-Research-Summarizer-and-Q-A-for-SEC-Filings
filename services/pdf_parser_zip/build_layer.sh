#!/usr/bin/env bash
set -euo pipefail

PY_VERSION="3.11"
LAYER_DIR="layer"

rm -rf "${LAYER_DIR}"
mkdir -p "${LAYER_DIR}/python"

# Build dependencies inside the Lambda Python base image (Linux-compatible wheels),
# but OVERRIDE the Lambda entrypoint (otherwise it expects a handler name).
docker run --rm \
  --entrypoint /bin/bash \
  -v "$(pwd)/${LAYER_DIR}:/opt" \
  -v "$(pwd)/requirements.txt:/tmp/requirements.txt" \
  public.ecr.aws/lambda/python:${PY_VERSION} \
  -lc "pip install --no-cache-dir -r /tmp/requirements.txt -t /opt/python"

# Zip the layer
cd "${LAYER_DIR}"
zip -r ../pdf_parser_deps_layer.zip python > /dev/null
cd ..

echo "Built layer zip: pdf_parser_deps_layer.zip"
du -sh "${LAYER_DIR}" || true