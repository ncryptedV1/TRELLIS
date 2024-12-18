#!/bin/bash
set -e

# Run post-install script if not already done
if [ ! -f /app/.post_install_done ]; then
  echo "Running post-install script..."
  /bin/bash /app/post_install.sh
fi

echo "Starting TRELLIS Gradio Demo..."

# Start the application
python /app/app.py
