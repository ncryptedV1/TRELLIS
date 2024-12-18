#!/bin/bash
set -e

# Check if post-install steps have already been run
if [ -f /app/.post_install_done ]; then
    echo "Post-install steps already completed."
    exit 0
fi

cd /app

echo "Installing GPU-dependent packages..."

# Run the setup script for GPU-dependent packages
/bin/bash ./setup.sh --diffoctreerast --mipgaussian

# Mark completion
touch /app/.post_install_done

echo "Post-install steps completed successfully."
