#!/usr/bin/env bash
set -euo pipefail

REGION="ap-southeast-1"
FUNC_NAME="sec-rag-part10a-openai-answer"
ROLE_NAME="sec-rag-part10a-openai-answer-role"
POLICY_NAME="sec-rag-part10a-openai-answer-policy"
BUCKET="sec-rag-ai-system"

: "${OPENAI_API_KEY:?Need OPENAI_API_KEY}"

OPENAI_MODEL_ANSWER="${OPENAI_MODEL_ANSWER:-gpt-5.2}"
OPENAI_TIMEOUT_S="${OPENAI_TIMEOUT_S:-60}"

# context controls
MAX_CONTEXT_CHUNKS="${MAX_CONTEXT_CHUNKS:-12}"
MAX_CHARS_PER_CHUNK="${MAX_CHARS_PER_CHUNK:-1400}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# 1) Build ZIP
./scripts/build_zip_docker.sh

# 2) Create/update IAM role + inline policy
aws iam get-role --role-name "$ROLE_NAME" >/dev/null 2>&1 || \
  aws iam create-role \
    --role-name "$ROLE_NAME" \
    --assume-role-policy-document file://iam/trust-policy.json \
    >/dev/null

aws iam put-role-policy \
  --role-name "$ROLE_NAME" \
  --policy-name "$POLICY_NAME" \
  --policy-document file://iam/permissions-policy.json \
  >/dev/null

ROLE_ARN="$(aws iam get-role --role-name "$ROLE_NAME" --query 'Role.Arn' --output text)"

# IAM propagation guard (prevents "role cannot be assumed")
sleep 15

# 3) Create or update Lambda code
if aws lambda get-function --function-name "$FUNC_NAME" --region "$REGION" >/dev/null 2>&1; then
  aws lambda update-function-code \
    --function-name "$FUNC_NAME" \
    --region "$REGION" \
    --zip-file fileb://dist/function.zip \
    >/dev/null

  # Wait until update finishes (avoids 'Pending' conflicts)
  aws lambda wait function-updated-v2 --function-name "$FUNC_NAME" --region "$REGION"
else
  aws lambda create-function \
    --function-name "$FUNC_NAME" \
    --runtime python3.11 \
    --handler handler.lambda_handler \
    --role "$ROLE_ARN" \
    --timeout 300 \
    --memory-size 2048 \
    --zip-file fileb://dist/function.zip \
    --region "$REGION" \
    >/dev/null
fi

# 4) Set env vars (update timeout to 900 if you want)
aws lambda update-function-configuration \
  --function-name "$FUNC_NAME" \
  --region "$REGION" \
  --timeout 900 \
  --environment "Variables={
    BUCKET=$BUCKET,
    DERIVED_PREFIX=derived,
    PIPELINE_VERSION=v1.0.0,

    OPENAI_API_KEY=$OPENAI_API_KEY,
    OPENAI_MODEL_ANSWER=$OPENAI_MODEL_ANSWER,
    OPENAI_TIMEOUT_S=$OPENAI_TIMEOUT_S,

    MAX_CONTEXT_CHUNKS=$MAX_CONTEXT_CHUNKS,
    MAX_CHARS_PER_CHUNK=$MAX_CHARS_PER_CHUNK
  }" \
  >/dev/null

aws lambda wait function-updated-v2 --function-name "$FUNC_NAME" --region "$REGION"

echo "Deployed/Updated: $FUNC_NAME ($REGION)"