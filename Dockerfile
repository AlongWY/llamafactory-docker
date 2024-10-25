# The LLAMA FACTORY Dockerfile is used to construct LLAMA FACTORY image that
# can be directly used to run the OpenAI compatible server.

#################### LLAMA FACTORY ####################
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel AS llamafactory

WORKDIR /workspace

# install additional dependencies for LLAMA FACTORY api server
RUN apt-get update && apt-get install -y --no-install-recommends git && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.cache/pip \
    python --version && \
    pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl \
    https://github.com/AlongWY/deepspeed_wheels/releases/download/v0.15.3/deepspeed-0.15.3+cu121torch2.4-cp311-cp311-manylinux_2_28_x86_64.whl \
    https://github.com/flashinfer-ai/flashinfer/releases/download/v0.1.6/flashinfer-0.1.6+cu121torch2.4-cp311-cp311-linux_x86_64.whl \
    "unsloth[huggingface] @ git+https://github.com/unslothai/unsloth.git" \
    bitsandbytes optimum auto-gptq autoawq hqq \
    transformers_stream_generator \
    modelscope openai setuptools \
    nltk jieba rouge-chinese \
    tensorboard wandb \
    galore-torch badam \
    liger-kernel \
    vllm \
    "llamafactory==0.9.0"

ENTRYPOINT ["llamafactory-cli"]
#################### LLAMA FACTORY ####################
