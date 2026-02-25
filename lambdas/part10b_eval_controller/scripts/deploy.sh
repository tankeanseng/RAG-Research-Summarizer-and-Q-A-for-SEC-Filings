#!/usr/bin/env bash
set -euo pipefail

REGION="ap-southeast-1"
FUNC_NAME="sec-rag-part10b-eval-controller"
ROLE_NAME="sec-rag-part10b-eval-controller-role"
POLICY_NAME="sec-rag-part10b-eval-controller-policy"
BUCKET="sec-rag-ai-system"

: "${OPENAI_API_KEY:?Need OPENAI_API_KEY}"

OPENAI_MODEL_JUDGE="${OPENAI_MODEL_JUDGE:-gpt-5.2}"
OPENAI_MODEL_ANSWER="${OPENAI_MODEL_ANSWER:-gpt-5.2}"
OPENAI_TIMEOUT_S="${OPENAI_TIMEOUT_S:-60}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

chmod +x scripts/build_zip_docker.sh
./scripts/build_zip_docker.sh

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
sleep 15

if aws lambda get-function --function-name "$FUNC_NAME" --region "$REGION" >/dev/null 2>&1; then
  aws lambda update-function-code \
    --function-name "$FUNC_NAME" \
    --region "$REGION" \
    --zip-file fileb://dist/function.zip \
    >/dev/null
else
  aws lambda create-function \
    --function-name "$FUNC_NAME" \
    --runtime python3.11 \
    --handler handler.lambda_handler \
    --role "$ROLE_ARN" \
    --timeout 900 \
    --memory-size 2048 \
    --zip-file fileb://dist/function.zip \
    --region "$REGION" \
    >/dev/null
fi

aws lambda wait function-updated-v2 --function-name "$FUNC_NAME" --region "$REGION"

aws lambda update-function-configuration \
  --function-name "$FUNC_NAME" \
  --region "$REGION" \
  --timeout 900 \
  --environment "Variables={
    BUCKET=$BUCKET,
    DERIVED_PREFIX=derived,
    PIPELINE_VERSION=v1.0.0,

    OPENAI_API_KEY=$OPENAI_API_KEY,
    OPENAI_MODEL_JUDGE=$OPENAI_MODEL_JUDGE,
    OPENAI_MODEL_ANSWER=$OPENAI_MODEL_ANSWER,
    OPENAI_TIMEOUT_S=$OPENAI_TIMEOUT_S,

    RUN_JUDGE=true,
    MAX_RETRIES=3,

    MIN_CITATION_COVERAGE=0.80,
    THRESH_FAITHFULNESS=0.80,
    THRESH_ANSWER_RELEVANCY=0.70,
    THRESH_CONTEXT_PRECISION=0.60,
    THRESH_CONTEXT_UTILIZATION=0.60,

    MAX_CHARS_PER_CHUNK=1400
  }" \
  >/dev/null

aws lambda wait function-updated-v2 --function-name "$FUNC_NAME" --region "$REGION"

echo "Deployed/Updated: $FUNC_NAME ($REGION)"