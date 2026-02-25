#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_env.sh"

ROLE_ARN=$(aws iam get-role \
  --role-name sec-rag-question-preprocessor-role \
  --query 'Role.Arn' --output text)

# If function exists, update; else create.
if aws lambda get-function --function-name "$FUNC_NAME" --region "$AWS_REGION" >/dev/null 2>&1; then
  echo "Updating Lambda code to image: $ECR_URI"
  aws lambda update-function-code \
    --region "$AWS_REGION" \
    --function-name "$FUNC_NAME" \
    --image-uri "$ECR_URI" >/dev/null
else
  echo "Creating Lambda from image: $ECR_URI"
  aws lambda create-function \
    --region "$AWS_REGION" \
    --function-name "$FUNC_NAME" \
    --package-type Image \
    --code ImageUri="$ECR_URI" \
    --role "$ROLE_ARN" \
    --timeout 900 \
    --memory-size 4096 >/dev/null
fi

# Match architecture to linux/amd64 build -> x86_64
aws lambda update-function-configuration \
  --region "$AWS_REGION" \
  --function-name "$FUNC_NAME" \
  --architectures x86_64 >/dev/null

echo "Deployed Lambda: $FUNC_NAME"