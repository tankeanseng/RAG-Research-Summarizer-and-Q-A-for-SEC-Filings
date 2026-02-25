#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_env.sh"

aws ecr get-login-password --region "$AWS_REGION" \
| docker login --username AWS --password-stdin \
  "${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"