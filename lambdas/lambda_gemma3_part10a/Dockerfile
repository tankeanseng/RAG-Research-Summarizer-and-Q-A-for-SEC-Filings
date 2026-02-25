FROM amazonlinux:2023 AS builder
RUN dnf -y install git make gcc gcc-c++ cmake findutils && dnf clean all

WORKDIR /src
RUN git clone https://github.com/ggml-org/llama.cpp.git
WORKDIR /src/llama.cpp

# Portable build
RUN mkdir -p build && cd build && \
    cmake .. \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_SHARED_LIBS=OFF \
      -DGGML_NATIVE=OFF \
      -DGGML_LTO=OFF && \
    cmake --build . --parallel

RUN mkdir -p /out && \
    cp "$(find /src/llama.cpp/build -maxdepth 8 -type f -name llama-cli | head -n 1)" /out/llama-cli && \
    chmod +x /out/llama-cli

FROM public.ecr.aws/lambda/python:3.12
RUN dnf -y install libstdc++ libgomp && dnf clean all

RUN mkdir -p /opt/llama /opt/model
COPY --from=builder /out/llama-cli /opt/llama/llama-cli
RUN chmod +x /opt/llama/llama-cli

COPY model/gemma3_merged_q4_k_m.gguf /opt/model/gemma3_merged_q4_k_m.gguf
COPY app/handler.py ${LAMBDA_TASK_ROOT}/handler.py

ENV MODEL_GGUF_PATH=/opt/model/gemma3_merged_q4_k_m.gguf \
    LLAMA_BIN=/opt/llama/llama-cli \
    BUCKET=sec-rag-ai-system \
    DERIVED_PREFIX=derived \
    PIPELINE_VERSION=v1.0.0

CMD ["handler.lambda_handler"]