FROM public.ecr.aws/lambda/python:3.11

# Install deps
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

# Copy function code
COPY app.py ${LAMBDA_TASK_ROOT}/app.py

# Handler
CMD ["app.lambda_handler"]