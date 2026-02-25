FROM public.ecr.aws/lambda/python:3.11

# Writable cache dirs in Lambda
ENV HF_HOME=/tmp/hf \
    HF_HUB_CACHE=/tmp/hf/hub \
    TRANSFORMERS_CACHE=/tmp/hf/transformers \
    XDG_CACHE_HOME=/tmp \
    TOKENIZERS_PARALLELISM=false

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
 && python -m pip install --no-cache-dir -r requirements.txt \
 && python -m pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch==2.2.2+cpu

# Bake model into image (no runtime download)
ENV MODEL_DIR=/opt/models/ms-marco-MiniLM-L6-v2
RUN python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="cross-encoder/ms-marco-MiniLM-L6-v2",
    local_dir="/opt/models/ms-marco-MiniLM-L6-v2",
    local_dir_use_symlinks=False
)
print("Downloaded model to /opt/models/ms-marco-MiniLM-L6-v2")
PY

COPY app/ ${LAMBDA_TASK_ROOT}/app/
CMD [ "app.handler.lambda_handler" ]