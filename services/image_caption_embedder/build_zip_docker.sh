#!/usr/bin/env bash
set -euo pipefail

rm -rf build package.zip

docker run --rm \
  -v "$PWD":/var/task \
  --entrypoint /bin/bash \
  public.ecr.aws/lambda/python:3.11 \
  -lc '
    cd /var/task
    python -m pip install -U pip
    mkdir -p build
    python -m pip install -r requirements.txt -t build
    cp handler.py build/
    python zip_build.py
  '

echo "Built package.zip"
ls -lh package.zip