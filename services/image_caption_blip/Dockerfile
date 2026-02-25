FROM public.ecr.aws/lambda/python:3.11

ENV HF_HOME=/opt/hf
ENV HF_HUB_CACHE=/opt/hf/hub
ENV TRANSFORMERS_CACHE=/opt/hf/transformers

# Copy requirements and install
COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir --only-binary=:all: numpy==1.26.4 && \
    python -m pip install --no-cache-dir -r requirements.txt

# Copy app
COPY app.py ${LAMBDA_TASK_ROOT}/app.py

# Optional: pre-download model at build time (reduces cold start).
# This will make the image bigger but faster at runtime.
RUN python -c "from transformers import BlipProcessor, BlipForConditionalGeneration; \
m='Salesforce/blip-image-captioning-base'; \
BlipProcessor.from_pretrained(m); \
BlipForConditionalGeneration.from_pretrained(m, use_safetensors=True)"

CMD ["app.lambda_handler"]