FROM ubuntu:16.04

RUN apt-get update && apt-get install -y \
    build-essential \
    binutils \
    make \
    bzip2 \
    cmake \
    curl \
    git \
    g++ \
    libboost-all-dev \
    libbz2-dev \
    libfluidsynth-dev \
    libfreetype6-dev \
    libgme-dev \
    libgtk2.0-dev \
    libjpeg-dev \
    libopenal-dev \
    libpng-dev \
    libsdl2-dev \
    libwildmidi-dev \
    libzmq3-dev \
    nano \
    nasm \
    pkg-config \
    rsync \
    software-properties-common \
    sudo \
    tar \
    timidity \
    unzip \
    wget \
    locales \
    zlib1g-dev \
    libfltk1.3-dev \
    libxft-dev \
    libxinerama-dev \
    libjpeg-dev \
    libpng-dev \
    zlib1g-dev \
    xdg-utils \
    net-tools 

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh  

RUN pip install --upgrade pip
RUN conda install numpy pandas matplotlib networkx tqdm scikit-learn imageio
RUN pip install geohash2
RUN conda install pytorch torchvision cudatoolkit=10.0 faiss-gpu -c pytorch
RUN conda install -c dglteam dgl-cuda10.0
RUN pip install fastapi uvicorn

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

