FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ARG DOCKER_UID=1000
ARG DOCKER_GID=1000
ARG DOCKER_USER=user
ARG HOME=/home/${DOCKER_USER}

RUN groupadd -g $DOCKER_GID $DOCKER_USER \
    && useradd --uid $DOCKER_UID --gid $DOCKER_GID -m $DOCKER_USER \
    && echo "$DOCKER_USER ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

## Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV NVIDIA_VISIBLE_DEVICES=all

## NOTE: Uncomment this command to change apt source if you encounter connection issues in China mainland
# RUN sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list && \
#     sed -i s@/security.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list

########################################################
## Install dependencies
########################################################
RUN apt update && apt install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    ssh \
    x11-apps \
    mesa-utils \
    ninja-build \
    vulkan-tools \
    libglu1 \
    # ref: https://askubuntu.com/a/1072878
    libglib2.0-0 \
    # ref: https://stackoverflow.com/a/76778289
    libxrandr2 \
    # below for dmlab
    libsdl2-dev \
    libosmesa6 \
    && apt clean
RUN apt install -y -o Dpkg::Options::="--force-confold" sudo

USER ${DOCKER_USER}
WORKDIR ${HOME}

## Use bash instead of sh
SHELL ["/bin/bash", "-c"]

## Install uv
RUN curl -fsSL https://astral.sh/uv/install.sh | bash
ENV PATH=${HOME}/.local/bin:$PATH
RUN echo "source ${HOME}/fishrl/.venv/bin/activate" >> ${HOME}/.bashrc

## Install Claude Code
RUN curl -fsSL https://claude.ai/install.sh | bash

########################################################
## Finalize
########################################################
ENTRYPOINT [ "./entrypoint.sh" ]
WORKDIR ${HOME}/fishrl
CMD ["/bin/bash"]
