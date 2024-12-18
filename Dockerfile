# Start from the NVIDIA CUDA base image with Ubuntu 20.04 and CUDA 11.8
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# Set environment variables to prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install essential packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.8 \
        python3-pip \
        python3-setuptools \
        python3-dev \
        build-essential \
        git \
        wget \
        curl \
        cmake \
        ffmpeg \
        libgl1-mesa-dev \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        pkg-config \
        libvulkan1 \
    && rm -rf /var/lib/apt/lists/*

# Ensure 'python' and 'pip' point to Python 3.8
RUN ln -sf /usr/bin/python3.8 /usr/bin/python && ln -sf /usr/bin/pip3 /usr/bin/pip

# Upgrade pip
RUN pip install --upgrade pip

# Install PyTorch with CUDA 11.8 support
RUN pip install --no-cache-dir torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# Install basic Python packages
RUN pip install --no-cache-dir \
    numpy \
    scipy \
    imageio \
    Pillow \
    scikit-image \
    tqdm \
    matplotlib \
    opencv-python \
    typing_extensions \
    timm \
    einops \
    ninja \
    wheel \
    gradio \
    xformers==0.0.20

# Create g++ wrapper for JIT compilation
RUN echo '#!/usr/bin/env bash\nexec /usr/bin/g++ -I/usr/local/cuda/include -I/usr/local/cuda/include/crt "$@"\n' > /usr/local/bin/gxx-wrapper && \
    chmod +x /usr/local/bin/gxx-wrapper
ENV CXX=/usr/local/bin/gxx-wrapper

# Copy application files
WORKDIR /app
COPY . /app

# Create the /nonexistent directory and set ownership to UID and GID 65534
RUN mkdir -p /nonexistent && \
    chown -R 65534:65534 /nonexistent

# Change ownership of the /app directory and necessary Python directories to UID and GID 65534
RUN chown -R 65534:65534 /app
# RUN chown -R 65534:65534 /app /usr/local/lib/python3.8/dist-packages /usr/local/bin

# Switch to user with UID and GID 65534
USER 65534:65534

# Expose the port for Gradio
EXPOSE 7860

# Set environment variables as needed
ENV ATTN_BACKEND=flash-attn
ENV SPCONV_ALGO=native

# Set the entrypoint script
ENTRYPOINT ["/bin/bash", "/app/onstart.sh"]  