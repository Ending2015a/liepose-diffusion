ARG BASE_IMAGE=nvidia/cuda:12.2.2-cudnn8-devel-ubuntu20.04

FROM $BASE_IMAGE AS base

# set timezone
ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update
RUN apt-get install -y --no-install-recommends \
    build-essential \
    git curl cmake autoconf automake libtool vim ca-certificates \
    libjpeg-dev libpng-dev libglfw3-dev libglm-dev libx11-dev libomp-dev \
    libegl1-mesa-dev pkg-config ffmpeg wget zip unzip g++ gcc \
    libosmesa6-dev python3-pip python3-dev \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

ADD lib/bop_renderer /bop_renderer
ADD lib/bop_toolkit /bop_toolkit

# PATH
ENV PYTHONPATH "${PYTHONPATH}:/bop_renderer/build"
ENV PATH "${PATH}:/bop_renderer/build"

RUN cd /bop_renderer && \
    cmake -B build -S . -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build

RUN pip install --upgrade pip

# Install bop_toolkit
RUN pip install /bop_toolkit

# Install dependencies
RUN pip install jax[cuda12_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

COPY ./requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

ARG WORKDIR=/workspace
WORKDIR ${WORKDIR}

COPY ./ ${WORKDIR}
RUN pip install -e ${WORKDIR}

CMD "/bin/bash"
