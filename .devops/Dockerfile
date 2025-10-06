FROM nvidia/cuda:13.0.1-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda-13.0
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git cmake ninja-build make \
        g++-14 gcc-14 pkg-config \
        libopenblas-dev \
        python3 python3-dev python3-pip \
        curl ca-certificates \
        patchelf rsync && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY requirements.txt .
COPY scripts/ /app/scripts/
RUN chmod +x /app/scripts/*.sh 

# Install FlashAttention & TorchVision before other deps
RUN /app/scripts/dependencies.sh

# Install Python dependencies
RUN uv pip install -r requirements.txt

# Build Python binding for C++ Engine
RUN /app/scripts/cuda_python_binding.sh

# TODO: pull models from HF

COPY src/ /app/src/
ENV PYTHONPATH=/app

EXPOSE 7860

CMD ["python3", "/app/src/run.py"]
