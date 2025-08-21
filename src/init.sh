#!/bin/bash

# tailscaled --tun=userspace-networking --socks5-server=localhost:1055 >/tmp/ts.log 2>&1 &
# tailscale up

pip install -U huggingface_hub hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1

hf download --repo-type model "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B" \
  --local-dir "/data/models/deepseek-r1-0528-qwen3-8b@gf16"

exec sleep infinity