#!/bin/bash
set -e

# Check if post-install steps have already been run
if [ -f /app/.post_install_done ]; then
    echo "Post-install steps already completed."
    exit 0
fi

cd /app

echo "Installing GPU-dependent base packages..."
/bin/bash ./setup.sh --basic 

echo "Testing ninja installation..."
ninja --version
echo $? # Exit code should be 0 if ninja is installed
# if ninja is not installed, installation of flash-attn will take 2h instead of 3-5min

echo "Installing GPU-dependent packages..."
# Run the setup script for GPU-dependent packages
/bin/bash ./setup.sh --xformers --flash-attn --vox2seq --spconv --kaolin --nvdiffrast --diffoctreerast --mipgaussian --demo

# Mark completion
touch /app/.post_install_done

echo "Post-install steps completed successfully."
