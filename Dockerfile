# The sglang Dockerfile is used to construct sglang image that
# can be directly used to run the OpenAI compatible server.

#################### SGLANG API SERVER ####################
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime AS sglang

WORKDIR /workspace

# install additional dependencies for sglang api server
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.9.post1/flash_attn-2.5.9.post1+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl \
    https://github.com/AlongWY/deepspeed_wheels/releases/download/v0.14.2/deepspeed-0.14.2+cu121torch2.3-cp310-cp310-manylinux_2_24_x86_64.whl \
    https://github.com/flashinfer-ai/flashinfer/releases/download/v0.0.4/flashinfer-0.0.4+cu121torch2.3-cp310-cp310-linux_x86_64.whl \
    https://github.com/vllm-project/vllm/releases/download/v0.4.3/vllm-0.4.3-cp310-cp310-manylinux1_x86_64.whl \
    git+https://github.com/unslothai/unsloth \
    bitsandbytes  optimum auto-gptq autoawq \
    transformers_stream_generator \
    modelscope openai setuptools \
    nltk jieba rouge-chinese \
    tensorboard wandb mlflow \
    galore-torch badam \
    protobuf==4.25.3 \
    "aqlm[gpu]" \
    llamafactory

ENTRYPOINT ["llamafactory-cli"]
#################### SGLANG API SERVER ####################
