FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

# Install basic utilities
RUN apt update && apt install nmap curl nano rsync -y

# Install & setup tailscale
RUN curl -fsSL https://tailscale.com/install.sh | sh
# todo: fix this
RUN tailscaled --tun=userspace-networking --socks5-server=localhost:1055 >/tmp/ts.log 2>&1 &


# setup workspace
ENV PV=/workspace

# redirect all caches/tmp to the volume
RUN mkdir -p $PV/.cache/{pip,huggingface,datasets,torch} $PV/tmp
ENV PIP_CACHE_DIR=$PV/.cache/pip
ENV HF_HOME=$PV/.cache/huggingface
ENV TRANSFORMERS_CACHE=$PV/.cache/huggingface/hub
ENV HF_DATASETS_CACHE=$PV/.cache/datasets
ENV TORCH_HOME=$PV/.cache/torch
ENV TMPDIR=$PV/tmp

# symlink default cache locations to the volume
RUN mkdir -p ~/.cache
RUN rm -rf ~/.cache/{pip,huggingface,torch} 2>/dev/null
RUN ln -s $PV/.cache/pip        ~/.cache/pip
RUN ln -s $PV/.cache/huggingface ~/.cache/huggingface
RUN ln -s $PV/.cache/torch      ~/.cache/torch


# configure bashrc
RUN cat >> ~/.bashrc <<'EOF'
export PV=/workspace
export PIP_CACHE_DIR=$PV/.cache/pip
export HF_HOME=$PV/.cache/huggingface
export TRANSFORMERS_CACHE=$PV/.cache/huggingface/hub
export HF_DATASETS_CACHE=$PV/.cache/datasets
export TORCH_HOME=$PV/.cache/torch
export TMPDIR=$PV/tmp
export ALL_PROXY=socks5://127.0.0.1:1055
. $PV/venv/bin/activate
EOF

# configure venv & install dependencies
RUN python3 -m venv $PV/venv
RUN source $PV/venv/bin/activate
RUN pip install --upgrade pip
RUN python3 -m pip install -U pip setuptools wheel pysocks
RUN export PIP_BREAK_SYSTEM_PACKAGES=1 # optional, in case of blinker issue
RUN pip install -U --ignore-installed blinker

# todo: go about this differently
COPY requirements.txt .
RUN pip install -r requirements.txt


# clean up
RUN pip cache purge
RUN rm -rf ~/.cache/* /root/.cache/* $HOME/.huggingface/* 2>/dev/null






