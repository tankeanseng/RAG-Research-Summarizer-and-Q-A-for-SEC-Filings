#!/usr/bin/env bash
set -euo pipefail

REGION="ap-southeast-1"
FUNC_NAME="sec-rag-part9a-doc-injection"
ROLE_NAME="sec-rag-part9a-doc-injection-role"
POLICY_NAME="sec-rag-part9a-doc-injection-policy"
BUCKET="sec-rag-ai-system"

: "${OPENAI_API_KEY:?Need OPENAI_API_KEY}"
OPENAI_MODEL_VERIFY="${OPENAI_MODEL_VERIFY:-gpt-5.2}"

cd "$(dirname "$0")/.."

# 1) Build zip
./scripts/build_zip_docker.sh

# 2) Create/update IAM role
aws iam get-role --role-name "$ROLE_NAME" >/dev/null 2>&1 || \
  aws iam create-role --role-name "$ROLE_NAME" --assume-role-policy-document file://iam/trust-policy.json >/dev/null

aws iam put-role-policy --role-name "$ROLE_NAME" --policy-name "$POLICY_NAME" --policy-document file://iam/permissions-policy.json >/dev/null
ROLE_ARN="$(aws iam get-role --role-name "$ROLE_NAME" --query 'Role.Arn' --output text)"

# Wait for IAM propagation (prevents "role cannot be assumed")
sleep 15

# 3) Create or update Lambda code
if aws lambda get-function --function-name "$FUNC_NAME" --region "$REGION" >/dev/null 2>&1; then
  aws lambda update-function-code \
    --function-name "$FUNC_NAME" \
    --region "$REGION" \
    --zip-file fileb://dist/function.zip >/dev/null
else
  aws lambda create-function \
    --function-name "$FUNC_NAME" \
    --runtime python3.11 \
    --handler handler.lambda_handler \
    --role "$ROLE_ARN" \
    --timeout 120 \
    --memory-size 1024 \
    --zip-file fileb://dist/function.zip \
    --region "$REGION" >/dev/null
fi

# 4) Update environment variables
aws lambda update-function-configuration \
  --function-name "$FUNC_NAME" --region "$REGION" \
  --environment "Variables={
    BUCKET=$BUCKET,
    DERIVED_PREFIX=derived,
    PIPELINE_VERSION=v1.0.0,
    OPENAI_API_KEY=$OPENAI_API_KEY,
    OPENAI_MODEL_VERIFY=$OPENAI_MODEL_VERIFY,
    OPENAI_TIMEOUT_S=45,
    TOP_K_SCAN=20
  }" >/dev/null

echo "Deployed/Updated: $FUNC_NAME"