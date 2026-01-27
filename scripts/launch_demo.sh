#!/bin/bash
# scripts/launch_demo.sh

# Function to display help
usage() {
    echo "Usage: $0 [mode]"
    echo "Modes:"
    echo "  local   - Run the demo locally (requires python env)"
    echo "  docker  - Build and run in Docker container"
    echo "  stop    - Stop the Docker container"
    exit 1
}

MODE=${1:-local}

if [ "$MODE" == "local" ]; then
    echo "🚀 Launching Demo Locally..."
    export PYTHONPATH=$PYTHONPATH:$(pwd)
    python -m demo.app

elif [ "$MODE" == "docker" ]; then
    echo "🐳 Building and Launching Docker Container..."
    docker build -t dholuo-tts-demo -f docker/Dockerfile.demo .
    
    echo "🏃 Running Container on port 7860..."
    docker run --rm -it \
        -p 7860:7860 \
        --name dholuo-demo \
        dholuo-tts-demo

elif [ "$MODE" == "stop" ]; then
    echo "🛑 Stopping Container..."
    docker stop dholuo-demo

else
    usage
fi
