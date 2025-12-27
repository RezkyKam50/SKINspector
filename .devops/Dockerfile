# this dockerfile is only for deployment and not for training.
# for training, we can directly leverage existing scripts on parents folder.

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
COPY models/ /app/models/
COPY src/ /app/src/

RUN chmod +x /app/scripts/*.sh 
RUN chmod +x /app/models/*.sh 

# Install FlashAttention & TorchVision before other deps
RUN /app/scripts/dependencies.sh

# Install Python dependencies
RUN uv pip install -r requirements.txt

# Build Python binding for C++ Engine
RUN /app/scripts/cuda_python_binding.sh

# Pull Vanilla model from HF
RUN /app/models/install_vlm.sh

ENV PYTHONPATH=/app

EXPOSE 7860

CMD ["python3", "/app/src/run.py"]
