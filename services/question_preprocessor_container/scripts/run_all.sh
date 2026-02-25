#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"

"$DIR/01_ecr_login.sh"
"$DIR/02_build_push.sh"
"$DIR/03_verify_manifest.sh"
"$DIR/04_deploy_lambda.sh"