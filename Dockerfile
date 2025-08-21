FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

# Install basic utilities
RUN apt update && apt install nmap curl nano rsync -y

# Install & setup tailscale
RUN curl -fsSL https://tailscale.com/install.sh | sh

#ENV PV=/workspace
WORKDIR /workspace

# redirect all caches/tmp to the volume
#RUN mkdir -p $PV/.cache/{pip,huggingface,datasets,torch} $PV/tmp
#ENV PIP_CACHE_DIR=$PV/.cache/pip
#ENV HF_HOME=$PV/.cache/huggingface
#ENV HF_DATASETS_CACHE=$PV/.cache/datasets
#ENV TORCH_HOME=$PV/.cache/torch
#ENV TMPDIR=$PV/tmp

## symlink default cache locations to the volume
#RUN mkdir -p ~/.cache
#RUN rm -rf ~/.cache/{pip,huggingface,torch} 2>/dev/null
#RUN ln -s $PV/.cache/pip        ~/.cache/pip
#RUN ln -s $PV/.cache/huggingface ~/.cache/huggingface
#RUN ln -s $PV/.cache/torch      ~/.cache/torch


# export PV=/workspace
# export PIP_CACHE_DIR=$PV/.cache/pip

# configure bashrc
# todo: remove superfluous stuff
RUN cat >> ~/.bashrc <<'EOF'
export HF_HOME=/data/.cache/huggingface
export HF_DATASETS_CACHE=/data/.cache/datasets
export TORCH_HOME=/data/.cache/torch
export TMPDIR=/data/tmp
# export ALL_PROXY=socks5://127.0.0.1:1055
. venv/bin/activate
EOF

# configure venv
RUN python3 -m venv venv
ENV VIRTUAL_ENV=/workspace/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# install deps
RUN pip install --upgrade pip
RUN python3 -m pip install -U pip setuptools wheel pysocks
RUN export PIP_BREAK_SYSTEM_PACKAGES=1 # optional, in case of blinker issue
RUN pip install -U --ignore-installed blinker

COPY requirements.txt .
RUN pip install -r requirements.txt
COPY .env  .
COPY src/ ./src/

# clean up
RUN pip cache purge
RUN rm -rf ~/.cache/* /root/.cache/* $HOME/.huggingface/* 2>/dev/null

# init
COPY src/init.sh /usr/local/bin/init.sh
RUN chmod +x /usr/local/bin/init.sh
ENTRYPOINT ["/usr/local/bin/init.sh"]





