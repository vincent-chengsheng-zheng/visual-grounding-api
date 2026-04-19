FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3.11-venv python3-pip git \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && rm -rf /var/lib/apt/lists/*

# PyTorch with CUDA 12.8 (pinned together to avoid version mismatch)
RUN pip3 install --no-cache-dir \
    torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128

# Constraints prevent transitive deps from upgrading torch to a CUDA-13 build,
# which would mismatch the cu128 torchvision installed above.
RUN printf "torch==2.11.0+cu128\ntorchvision==0.26.0+cu128\n" > /tmp/torch-constraints.txt

# Project dependencies
COPY requirements.txt .
RUN pip3 install -r requirements.txt --no-cache-dir -c /tmp/torch-constraints.txt
RUN pip3 install flash-attn --no-cache-dir --no-build-isolation -c /tmp/torch-constraints.txt
# Override the bitsandbytes==0.41.1 pin in requirements.txt — that version
# predates CUDA 12.x support and has no libbitsandbytes_cuda128.so.
RUN pip3 install --no-cache-dir --upgrade "bitsandbytes>=0.45.0" -c /tmp/torch-constraints.txt

# Copy source code
COPY core/ core/
COPY api/ api/
COPY pipeline/ pipeline/
COPY demo/ demo/
COPY shared/ shared/
COPY data/*.pkl data/
COPY data/dataset_stats.json data/
COPY setup.cfg .

# Weights and checkpoints mounted at runtime:
#   -v /path/to/weights:/app/weights
#   -v /path/to/checkpoints:/app/checkpoints

ENV TRANSFORMERS_OFFLINE=1
ENV LOAD_MODE=sequential

EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
