#!/bin/bash  
set -e  
  
echo "Running post-install script..."  
/bin/bash /app/post_install.sh  
  
echo "Starting TRELLIS Gradio Demo..."  
  
# Start the application  
python /app/app.py  
