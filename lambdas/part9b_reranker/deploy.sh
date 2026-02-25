#!/usr/bin/env bash
set -euo pipefail

AWS_REGION="ap-southeast-1"
BUCKET="sec-rag-ai-system"

ECR_REPO="sec-rag-part9b-reranker"
IMAGE_TAG="v4"   # bump each redeploy

FUNCTION_NAME="sec-rag-part9b-reranker"
ROLE_NAME="sec-rag-part9b-reranker-role"
POLICY_NAME="sec-rag-part9b-reranker-s3-policy"

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
echo "Account: ${ACCOUNT_ID}"
echo "Region:  ${AWS_REGION}"

# 1) ECR repo
aws ecr describe-repositories --repository-names "${ECR_REPO}" --region "${AWS_REGION}" >/dev/null 2>&1 || \
  aws ecr create-repository --repository-name "${ECR_REPO}" --region "${AWS_REGION}" >/dev/null

ECR_URI="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:${IMAGE_TAG}"

# 2) ECR login
aws ecr get-login-password --region "${AWS_REGION}" | \
  docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

# 3) Build image (avoid OCI index/manifest-list surprises)
docker buildx create --use >/dev/null 2>&1 || true
docker buildx build \
  --platform linux/amd64 \
  --provenance=false \
  --sbom=false \
  --output=type=docker \
  -t "${ECR_URI}" \
  .

# 4) Push
docker push "${ECR_URI}"

# 5) IAM role
aws iam get-role --role-name "${ROLE_NAME}" >/dev/null 2>&1 || \
  aws iam create-role \
    --role-name "${ROLE_NAME}" \
    --assume-role-policy-document '{
      "Version":"2012-10-17",
      "Statement":[{
        "Effect":"Allow",
        "Principal":{"Service":"lambda.amazonaws.com"},
        "Action":"sts:AssumeRole"
      }]
    }' >/dev/null

ROLE_ARN="$(aws iam get-role --role-name "${ROLE_NAME}" --query Role.Arn --output text)"

# 6) Basic logs policy
aws iam attach-role-policy \
  --role-name "${ROLE_NAME}" \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole >/dev/null

# 7) S3 policy attach/update
POLICY_ARN="$(aws iam list-policies --scope Local --query "Policies[?PolicyName=='${POLICY_NAME}'].Arn | [0]" --output text)"
if [[ "${POLICY_ARN}" == "None" || -z "${POLICY_ARN}" ]]; then
  POLICY_ARN="$(aws iam create-policy --policy-name "${POLICY_NAME}" --policy-document file://iam-policy-s3-part9b.json --query Policy.Arn --output text)"
else
  aws iam create-policy-version \
    --policy-arn "${POLICY_ARN}" \
    --policy-document file://iam-policy-s3-part9b.json \
    --set-as-default >/dev/null
fi
aws iam attach-role-policy --role-name "${ROLE_NAME}" --policy-arn "${POLICY_ARN}" >/dev/null

sleep 8

# 8) Create/update Lambda
if aws lambda get-function --function-name "${FUNCTION_NAME}" --region "${AWS_REGION}" >/dev/null 2>&1; then
  echo "Updating Lambda code..."
  aws lambda update-function-code \
    --function-name "${FUNCTION_NAME}" \
    --image-uri "${ECR_URI}" \
    --region "${AWS_REGION}" >/dev/null

  aws lambda wait function-updated-v2 \
    --function-name "${FUNCTION_NAME}" \
    --region "${AWS_REGION}"
else
  echo "Creating Lambda function..."
  aws lambda create-function \
    --function-name "${FUNCTION_NAME}" \
    --package-type Image \
    --code ImageUri="${ECR_URI}" \
    --role "${ROLE_ARN}" \
    --timeout 900 \
    --memory-size 2048 \
    --region "${AWS_REGION}" >/dev/null

  aws lambda wait function-active-v2 \
    --function-name "${FUNCTION_NAME}" \
    --region "${AWS_REGION}"
fi

# 9) Config (timeout=900)
aws lambda update-function-configuration \
  --function-name "${FUNCTION_NAME}" \
  --timeout 900 \
  --memory-size 2048 \
  --environment "Variables={BUCKET=${BUCKET},TOP_N=12,BATCH_SIZE=16,MODEL_DIR=/opt/models/ms-marco-MiniLM-L6-v2,HF_HOME=/tmp/hf,HF_HUB_CACHE=/tmp/hf/hub,TRANSFORMERS_CACHE=/tmp/hf/transformers,XDG_CACHE_HOME=/tmp,TOKENIZERS_PARALLELISM=false}" \
  --region "${AWS_REGION}" >/dev/null

aws lambda wait function-updated-v2 \
  --function-name "${FUNCTION_NAME}" \
  --region "${AWS_REGION}"

echo "✅ Deployed: ${FUNCTION_NAME}"
echo "   Image: ${ECR_URI}"