#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_env.sh"

# Ensure repo exists
aws ecr describe-repositories --repository-names "$REPO" --region "$AWS_REGION" >/dev/null 2>&1 \
  || aws ecr create-repository --repository-name "$REPO" --region "$AWS_REGION" >/dev/null

# Ensure a builder exists
docker buildx create --name lambda_builder --use >/dev/null 2>&1 || docker buildx use lambda_builder
docker buildx inspect --bootstrap >/dev/null

# IMPORTANT:
# - Use single arch
# - Disable provenance/SBOM
# - Force Docker media types: oci-mediatypes=false
# Docker docs: image/registry exporter supports oci-mediatypes=true|false. :contentReference[oaicite:2]{index=2}
docker buildx build \
  --platform linux/amd64 \
  --provenance=false \
  --sbom=false \
  -t "$ECR_URI" \
  -o type=registry,name="$ECR_URI",oci-mediatypes=false \
  .

echo "Built & pushed with Docker media types: $ECR_URI"