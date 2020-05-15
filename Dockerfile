FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu16.04

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
    python3 \
    python3-pip \
    net-tools 


RUN pip3 install --upgrade pip
RUN pip3 install numpy pandas matplotlib networkx geohash2 tqdm sklearn
RUN pip3 install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html 
RUN pip3 install dgl-cu101     # For CUDA 10.1 Build
