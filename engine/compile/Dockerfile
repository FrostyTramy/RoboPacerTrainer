# Hailo DFC 3.34.0 compilation environment (Ubuntu 22.04 + Python 3.10)
# Build context: D:\RoboPacer\engine\
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-dev \
        python3-pip \
        libgomp1 \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        wget \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m pip install --upgrade pip setuptools wheel

# Pre-install DFC runtime dependencies (avoids pip resolver timeouts inside Docker)
RUN pip install \
    "numpy>=1.23.5,<2.0" \
    scipy \
    matplotlib \
    pillow \
    networkx \
    flatbuffers \
    PyYAML \
    tqdm \
    onnx \
    onnxruntime \
    "tensorflow-cpu==2.12.0"

# Install Hailo Dataflow Compiler (Linux x86_64 wheel)
COPY resources/hailo_dataflow_compiler-3.34.0-py3-none-linux_x86_64.whl /tmp/
RUN pip install /tmp/hailo_dataflow_compiler-3.34.0-py3-none-linux_x86_64.whl

# Verify installation
RUN python3.10 -c "import hailo_sdk_client; print('DFC OK')"

WORKDIR /workspace
