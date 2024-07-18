# The LLAMA FACTORY Dockerfile is used to construct LLAMA FACTORY image that
# can be directly used to run the OpenAI compatible server.

#################### LLAMA FACTORY ####################
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime AS llamafactory

WORKDIR /workspace

# install additional dependencies for LLAMA FACTORY api server
RUN apt-get update && apt-get install -y --no-install-recommends git && apt-get clean && rm -rf /var/lib/apt/lists/*
    
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.1/flash_attn-2.6.1+cu123torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl \
    https://github.com/AlongWY/deepspeed_wheels/releases/download/v0.14.4/deepspeed-0.14.4+cu121torch2.3-cp310-cp310-manylinux_2_24_x86_64.whl \
    https://github.com/flashinfer-ai/flashinfer/releases/download/v0.1.0/flashinfer-0.1.0+cu121torch2.3-cp310-cp310-linux_x86_64.whl \
    https://github.com/vllm-project/vllm/releases/download/v0.5.2/vllm-0.5.2-cp310-cp310-manylinux1_x86_64.whl \
    "unsloth[cu121] @ git+https://github.com/unslothai/unsloth.git" \
    bitsandbytes optimum auto-gptq autoawq hqq eetq awq "aqlm[gpu,cpu]>=1.1.0" \
    transformers_stream_generator \
    modelscope openai setuptools \
    nltk jieba rouge-chinese \
    tensorboard wandb mlflow \
    galore-torch badam \
    protobuf==4.25.3 \
    "llamafactory @ git+https://github.com/hiyouga/LLaMA-Factory.git"

ENTRYPOINT ["llamafactory-cli"]
#################### LLAMA FACTORY ####################
