#!/usr/bin/env bash
set -euo pipefail

export AWS_REGION="ap-southeast-1"
export ACCOUNT_ID="087356279741"
export REPO="sec-rag-question-preprocessor"
export TAG="lambda"   # use a dedicated tag for Lambda images
export FUNC_NAME="sec-rag-question-preprocessor"

export ECR_URI="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${REPO}:${TAG}"

# IMPORTANT: Disable default attestations in buildx (helps avoid OCI index / extra artifacts)
# Docker docs: provenance can be disabled via --provenance=false or BUILDX_NO_DEFAULT_ATTESTATIONS env var. :contentReference[oaicite:2]{index=2}
export BUILDX_NO_DEFAULT_ATTESTATIONS="1"