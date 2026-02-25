#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_env.sh"

MEDIA=$(aws ecr batch-get-image \
  --region "$AWS_REGION" \
  --repository-name "$REPO" \
  --image-ids imageTag="$TAG" \
  --query 'images[0].imageManifestMediaType' \
  --output text)

echo "ECR imageManifestMediaType: $MEDIA"

if [[ "$MEDIA" != "application/vnd.docker.distribution.manifest.v2+json" ]]; then
  echo "ERROR: Lambda will reject this media type."
  echo "Expected: application/vnd.docker.distribution.manifest.v2+json"
  echo "Got:      $MEDIA"
  echo ""
  echo "Re-run scripts/02_build_push.sh. If still OCI index, your Docker/Buildx may be forcing OCI output."
  exit 1
fi

echo "OK: Manifest is Lambda-compatible."