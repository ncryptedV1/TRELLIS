# Start from the NVIDIA CUDA base image with Ubuntu 22.04 and CUDA 11.8
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment variables to prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install essential packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-setuptools \
    python3-dev \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Ensure 'python' and 'pip' point to Python 3.10
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && ln -sf /usr/bin/pip3 /usr/bin/pip

# Upgrade pip
RUN pip install --upgrade pip

# Install PyTorch with CUDA 11.8 support
RUN pip install --no-cache-dir 'numpy<2' torch==2.4.0+cu118 torchvision==0.19.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# Create g++ wrapper for JIT compilation
RUN echo '#!/usr/bin/env bash\nexec /usr/bin/g++ -I/usr/local/cuda/include -I/usr/local/cuda/include/crt "$@"\n' > /usr/local/bin/gxx-wrapper && \
    chmod +x /usr/local/bin/gxx-wrapper
ENV CXX=/usr/local/bin/gxx-wrapper

# Copy setup.sh
WORKDIR /app
COPY setup.sh /app/setup.sh

# Run setup.sh - this won't install all the things due to missing GPU in the builder
RUN ./setup.sh --basic --xformers --flash-attn --vox2seq --spconv --nvdiffrast --demo

# Copy these last, so we can experiment without excessive build times.
COPY trellis         /app/trellis
COPY app.py          /app/app.py
COPY example.py      /app/example.py
COPY extensions      /app/extensions
COPY assets          /app/assets
COPY onstart.sh      /app/onstart.sh
COPY post_install.sh /app/post_install.sh

# Create the /nonexistent (home) directory and set ownership to UID and GID 65534 (AI Core-specific)
RUN mkdir -p /nonexistent && \
    chown -R 65534:65534 /nonexistent

# Change ownership of the /app directory
RUN chown -R 65534:65534 /app
# ...and necessary Python directories to UID and GID 65534 (not required as pip falls back to user directory install)
# RUN chown -R 65534:65534 /usr/local/lib/python3.10/dist-packages /usr/local/bin

# Switch to user with UID and GID 65534
USER 65534:65534

# Expose the port for Gradio
EXPOSE 7860

# Set environment variables as needed
ENV ATTN_BACKEND=flash-attn
ENV SPCONV_ALGO=native

# This script runs the post_install steps

# If you're pushing to a container registry, let this run once, run some
# tests, then do `docker commit` to save the models along with the image.
# This will ensure that it won't fail at runtime due to models being
# unavailable, or network restrictions.
CMD ["/bin/bash", "/app/onstart.sh"]  