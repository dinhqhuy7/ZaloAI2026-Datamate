# Inference image (offline): vLLM + DataMate. Build from the repo ROOT:
#   docker build -f docker/Dockerfile -t datamate .
# Two folders that are not in the repo must be prepared first (see docker/README.md):
#   docker/models/qwen          Qwen3.5-9B (LoRA merged if used)   -> /models/qwen
#   docker/models/qwen_norm4b   Qwen3.5-4B (LoRA merged if used)   -> /models/qwen_norm4b
#   docker/assets/              few-shot example store (examples_*.jsonl, readme_train/)  -> /app/assets
FROM vllm/vllm-openai:v0.17.0

# the vLLM image has no `python` command, only python3; the base image sets ENTRYPOINT to vllm, so reset it
ENTRYPOINT []
ENV PY=python3 \
    PYTHONPATH=/app \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    NCCL_P2P_DISABLE=1

WORKDIR /app
# exactly the versions used during development and measurement (same as requirements.txt at the repo root): other versions may change SQL results or how files are read
RUN python3 -m pip install duckdb==1.5.6 pandas==2.3.3 python-calamine==0.8.3

COPY docker/models/qwen /models/qwen
COPY docker/models/qwen_norm4b /models/qwen_norm4b
COPY datamate /app/datamate
COPY docker/assets /app/assets
COPY docker/run.sh /app/run.sh
COPY docker/predict.sh /code/predict.sh

# a file written on Windows may carry a trailing \r: bash inside the container cannot run it
RUN sed -i 's/\r$//' /app/run.sh /code/predict.sh \
 && chmod +x /app/run.sh /code/predict.sh \
 && python3 -c "import vllm, duckdb, pandas, python_calamine; print('import ok')" \
 && test -f /models/qwen/config.json && test -f /models/qwen_norm4b/config.json \
 && test -f /app/assets/examples_train.jsonl

WORKDIR /code
CMD ["bash", "/code/predict.sh"]
