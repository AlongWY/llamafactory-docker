# The LLAMA FACTORY Dockerfile is used to construct LLAMA FACTORY image that
# can be directly used to run the OpenAI compatible server.

#################### LLAMA FACTORY ####################
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime AS LLAMA FACTORY

WORKDIR /workspace

# install additional dependencies for LLAMA FACTORY api server
RUN --mount=type=cache,target=/root/.cache/pip \
    apt-get update && apt-get install -y --no-install-recommends git && apt-get clean && rm -rf /var/lib/apt/lists/* \
    pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.9.post1/flash_attn-2.5.9.post1+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl \
    https://github.com/AlongWY/deepspeed_wheels/releases/download/v0.14.4/deepspeed-0.14.4+cu121torch2.3-cp310-cp310-manylinux_2_24_x86_64.whl \
    https://github.com/flashinfer-ai/flashinfer/releases/download/v0.0.6/flashinfer-0.0.6+cu121torch2.3-cp310-cp310-linux_x86_64.whl \
    https://github.com/vllm-project/vllm/releases/download/v0.5.0.post1/vllm-0.5.0.post1-cp310-cp310-manylinux1_x86_64.whl \
    "unsloth[cu121] @ git+https://github.com/unslothai/unsloth.git" \
    bitsandbytes optimum auto-gptq autoawq \
    transformers_stream_generator \
    modelscope openai setuptools \
    nltk jieba rouge-chinese \
    tensorboard wandb mlflow \
    galore-torch badam \
    protobuf==4.25.3 \
    "aqlm[gpu]" \
    llamafactory

ENTRYPOINT ["llamafactory-cli"]
#################### LLAMA FACTORY ####################
