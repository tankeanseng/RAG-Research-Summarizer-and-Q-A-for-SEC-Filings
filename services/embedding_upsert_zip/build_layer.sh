#!/usr/bin/env bash
set -euo pipefail

PY_VERSION="3.11"
LAYER_DIR="layer"

rm -rf "${LAYER_DIR}"
mkdir -p "${LAYER_DIR}/python"

docker run --rm \
  --entrypoint /bin/bash \
  -v "$(pwd)/${LAYER_DIR}:/opt" \
  -v "$(pwd)/requirements.txt:/tmp/requirements.txt" \
  public.ecr.aws/lambda/python:${PY_VERSION} \
  -lc "pip install --no-cache-dir -r /tmp/requirements.txt -t /opt/python"

cd "${LAYER_DIR}"
zip -r ../sec_rag_embedder_deps_layer.zip python > /dev/null
cd ..

echo "Built: sec_rag_embedder_deps_layer.zip"
du -sh "${LAYER_DIR}" || true