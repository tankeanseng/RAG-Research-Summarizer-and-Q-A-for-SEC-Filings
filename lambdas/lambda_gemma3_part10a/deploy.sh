#!/usr/bin/env bash
set -euo pipefail

: "${AWS_REGION:=ap-southeast-1}"
: "${AWS_ACCOUNT_ID:=087356279741}"

: "${ECR_REPO:=sec-rag-part10a-gemma3}"
: "${LAMBDA_FN:=sec-rag-part10a-gemma3}"
: "${ROLE_NAME:=sec-rag-part10a-gemma3-role}"
: "${INLINE_POLICY_NAME:=sec-rag-part10a-gemma3-policy}"

: "${TIMEOUT_SECONDS:=900}"
: "${MEMORY_MB:=10240}"
: "${ARCH:=x86_64}"

: "${S3_BUCKET:=sec-rag-ai-system}"
: "${DERIVED_PREFIX:=derived}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IAM_TRUST_FILE="${SCRIPT_DIR}/iam_trust.json"
IAM_POLICY_FILE="${SCRIPT_DIR}/iam_policy.json"

IMAGE_TAG="v$(date +%Y%m%d-%H%M%S)"
ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
IMAGE_URI="${ECR_URI}/${ECR_REPO}:${IMAGE_TAG}"

command -v aws >/dev/null 2>&1 || { echo "ERROR: aws cli not found"; exit 1; }
command -v docker >/dev/null 2>&1 || { echo "ERROR: docker not found"; exit 1; }

[[ -f "${IAM_TRUST_FILE}" ]] || { echo "ERROR: missing ${IAM_TRUST_FILE}"; exit 1; }
[[ -f "${IAM_POLICY_FILE}" ]] || { echo "ERROR: missing ${IAM_POLICY_FILE}"; exit 1; }

echo "=== Deploy Gemma3 Lambda ==="
echo "Image:  ${IMAGE_URI}"
echo "Lambda: ${LAMBDA_FN}"

aws ecr describe-repositories --region "${AWS_REGION}" --repository-names "${ECR_REPO}" >/dev/null 2>&1 \
  || aws ecr create-repository --region "${AWS_REGION}" --repository-name "${ECR_REPO}" >/dev/null

aws ecr get-login-password --region "${AWS_REGION}" \
| docker login --username AWS --password-stdin "${ECR_URI}" >/dev/null

docker buildx create --use --name lambda_builder >/dev/null 2>&1 || true
docker buildx build \
  --platform linux/amd64 \
  --provenance=false \
  --sbom=false \
  --load \
  -t "${ECR_REPO}:${IMAGE_TAG}" \
  "${SCRIPT_DIR}"

docker tag "${ECR_REPO}:${IMAGE_TAG}" "${IMAGE_URI}"
docker push "${IMAGE_URI}"

aws iam get-role --role-name "${ROLE_NAME}" >/dev/null 2>&1 || \
  aws iam create-role --role-name "${ROLE_NAME}" \
    --assume-role-policy-document "file://${IAM_TRUST_FILE}" >/dev/null

ROLE_ARN="$(aws iam get-role --role-name "${ROLE_NAME}" --query 'Role.Arn' --output text)"

aws iam put-role-policy \
  --role-name "${ROLE_NAME}" \
  --policy-name "${INLINE_POLICY_NAME}" \
  --policy-document "file://${IAM_POLICY_FILE}" >/dev/null

sleep 5

if aws lambda get-function --function-name "${LAMBDA_FN}" --region "${AWS_REGION}" >/dev/null 2>&1; then
  aws lambda update-function-code \
    --function-name "${LAMBDA_FN}" \
    --region "${AWS_REGION}" \
    --image-uri "${IMAGE_URI}" >/dev/null
  aws lambda wait function-updated-v2 --function-name "${LAMBDA_FN}" --region "${AWS_REGION}"
else
  aws lambda create-function \
    --function-name "${LAMBDA_FN}" \
    --package-type Image \
    --code ImageUri="${IMAGE_URI}" \
    --role "${ROLE_ARN}" \
    --timeout "${TIMEOUT_SECONDS}" \
    --memory-size "${MEMORY_MB}" \
    --architectures "${ARCH}" \
    --region "${AWS_REGION}" >/dev/null
  aws lambda wait function-active-v2 --function-name "${LAMBDA_FN}" --region "${AWS_REGION}"
fi

aws lambda update-function-configuration \
  --function-name "${LAMBDA_FN}" \
  --region "${AWS_REGION}" \
  --timeout "${TIMEOUT_SECONDS}" \
  --memory-size "${MEMORY_MB}" \
  --environment "Variables={BUCKET=${S3_BUCKET},DERIVED_PREFIX=${DERIVED_PREFIX},MODEL_GGUF_PATH=/opt/model/gemma3_merged_q4_k_m.gguf,LLAMA_BIN=/opt/llama/llama-cli,PIPELINE_VERSION=v1.0.0,MAX_CHARS_PER_CHUNK=600,MAX_QUESTION_CHARS=600,CTX_SIZE=768,MAX_NEW_TOKENS=140,N_BATCH=8,N_UBATCH=4,THREADS=4,TEMPERATURE=0.2,TOP_P=0.9,FIRST_OUTPUT_TIMEOUT_S=180,MIN_OUTPUT_CHARS_BEFORE_GUARDS=30,POST_OUTPUT_TIME_BUDGET_S=360,RSS_WATERMARK_MB=9000}" >/dev/null

aws lambda wait function-updated-v2 --function-name "${LAMBDA_FN}" --region "${AWS_REGION}"
echo "✅ Done. Deployed image: ${IMAGE_URI}"