# Setup script for Python 3.12 environment with vLLM installation
```bash
# Create new Python 3.12 environment
uv venv --python 3.12
# Activate the environment
source .venv/bin/activate
# Install base dependencies
uv sync
```
```bash
# Install vLLM separately with pre-release flag
uv pip install --pre vllm==0.10.1+gptoss \
    --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
    --extra-index-url https://download.pytorch.org/whl/nightly/cu128 \
    --index-strategy unsafe-best-match
```
```bash
# Alternative: If you want to manage vLLM in requirements.txt
echo "vllm==0.10.1+gptoss" > requirements-vllm.txt
uv pip install --pre -r requirements-vllm.txt \
    --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
    --extra-index-url https://download.pytorch.org/whl/nightly/cu128 \
    --index-strategy unsafe-best-match
```

# Create a .env file with the following content
```bash
# .env
VLLM_URL="http://localhost:8000/v1"
MODEL_ID="openai/gpt-oss-20b"
OPENAI_API_KEY="dummy"  # vLLM doesn't need a real key
VLLM_HOST="0.0.0.0"
VLLM_PORT="8000"
```

# Start vLLM server
```bash
# Basic server start (downloads model on first run)
vllm serve openai/gpt-oss-20b \
    --port 8000 \
    --async-scheduling
```
```bash
# Production configuration with optimizations
vllm serve openai/gpt-oss-20b \
    --host 0.0.0.0 \
    --port 8000 \
    --async-scheduling \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.95 \
    --enable-prefix-caching
```
```bash
# For debugging/testing with reduced memory
vllm serve openai/gpt-oss-20b \
    --port 8000 \
    --async-scheduling \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.9
```
```bash
# Using Docker (alternative)
docker run --gpus all \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:0.10.1 \
    --model openai/gpt-oss-20b \
    --async-scheduling
```




# Project Usage
