#!/bin/bash  
set -e  
  
# Check if post-install steps have already been run  
if [ -f /app/.post_install_done ]; then  
    echo "Post-install steps already completed."  
    exit 0  
fi  
  
echo "Installing GPU-dependent packages..."  
  
# Upgrade pip (in case it's outdated)  
pip install --upgrade pip  
  
# Install GPU-dependent packages  
pip install --no-cache-dir \  
    git+https://github.com/Dao-AILab/flash-attention.git@v1.0.4 \  
    spconv-cu118 \  
    'git+https://github.com/NVlabs/nvdiffrast.git' \  
    'git+https://github.com/JeffreyXiang/diffoctreerast.git' \  
    'git+https://github.com/trellis3d/mip-multiscale-gaussian.git' \  
    kaolin  
  
# Install TRELLIS package  
cd /app  
pip install --no-cache-dir -e .  
  
# Mark completion  
touch /app/.post_install_done  
  
echo "Post-install steps completed successfully."  
