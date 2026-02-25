#!/usr/bin/env bash
set -euo pipefail

REGION="ap-southeast-1"
FUNC_NAME="sec-rag-part8-retrieval"
ROLE_NAME="sec-rag-part8-retrieval-role"
POLICY_NAME="sec-rag-part8-retrieval-policy"
BUCKET="sec-rag-ai-system"

OPENAI_API_KEY="${OPENAI_API_KEY:-}"
PINECONE_API_KEY="${PINECONE_API_KEY:-}"
PINECONE_HOST="${PINECONE_HOST:-}"
PINECONE_NAMESPACE="${PINECONE_NAMESPACE:-sec}"

# embedding model: OpenAI
OPENAI_EMBED_MODEL="${OPENAI_EMBED_MODEL:-text-embedding-3-small}"
OPENAI_EMBED_DIMENSIONS="${OPENAI_EMBED_DIMENSIONS:-}"  # optional

# generation model for subqueries
OPENAI_MODEL_SUBQ="${OPENAI_MODEL_SUBQ:-gpt-5.2}"

if [[ -z "$OPENAI_API_KEY" || -z "$PINECONE_API_KEY" || -z "$PINECONE_HOST" ]]; then
  echo "Missing env vars. Export:"
  echo "  OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_HOST"
  exit 1
fi

cd "$(dirname "$0")/.."

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

if aws lambda get-function --function-name "$FUNC_NAME" --region "$REGION" >/dev/null 2>&1; then
  aws lambda update-function-code \
    --function-name "$FUNC_NAME" \
    --zip-file fileb://dist/function.zip \
    --region "$REGION" \
    >/dev/null
else
  aws lambda create-function \
    --function-name "$FUNC_NAME" \
    --runtime python3.11 \
    --handler handler.lambda_handler \
    --role "$ROLE_ARN" \
    --timeout 120 \
    --memory-size 2048 \
    --zip-file fileb://dist/function.zip \
    --region "$REGION" \
    >/dev/null
fi

# set env vars
if [[ -n "$OPENAI_EMBED_DIMENSIONS" ]]; then
  DIM_LINE="OPENAI_EMBED_DIMENSIONS=$OPENAI_EMBED_DIMENSIONS,"
else
  DIM_LINE=""
fi

aws lambda update-function-configuration \
  --function-name "$FUNC_NAME" \
  --region "$REGION" \
  --environment "Variables={
    BUCKET=$BUCKET,
    DERIVED_PREFIX=derived,
    PIPELINE_VERSION=v1.0.0,

    OPENAI_API_KEY=$OPENAI_API_KEY,
    OPENAI_TIMEOUT_S=45,
    OPENAI_MODEL_SUBQ=$OPENAI_MODEL_SUBQ,

    OPENAI_EMBED_MODEL=$OPENAI_EMBED_MODEL,
    $DIM_LINE

    PINECONE_API_KEY=$PINECONE_API_KEY,
    PINECONE_HOST=$PINECONE_HOST,
    PINECONE_NAMESPACE=$PINECONE_NAMESPACE,

    RETRIEVAL_TOP_K=20,
    RRF_K=60,
    MIN_TOP_DENSE_COSINE=0.35,
    MIN_TOP_RRF_SCORE=0.01,
    MIN_EVIDENCE_COUNT=3
  }" \
  >/dev/null

echo "Deployed: $FUNC_NAME ($REGION)"
echo "NOTE: Ensure Pinecone index dimension matches your embedding model. text-embedding-3-small defaults to 1536."