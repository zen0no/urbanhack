FROM ubuntu:22.04

WORKDIR /solution

# dependencies
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get upgrade -y

RUN apt-get install -y \
        build-essential git python3 python3-pip wget \
        ffmpeg libsm6 libxext6 libxrender1 libglib2.0-0

RUN pip3 install -U pip
RUN pip3 install --upgrade pip
RUN pip3 install gdown
RUN pip3 install torch==2.0.1 torchvision==0.15.2
RUN pip3 install albumentations==1.3.1
RUN pip3 install openmim==0.3.9 mmengine==0.8.4 pip yapf==0.40.1
RUN mim install mmcv==2.0.1 mmdet==3.1.0 mmyolo==0.6.0 mmpretrain==1.0.2

COPY ./config  ./config
COPY ./scripts ./scripts

# model weights
RUN mkdir -p ./weights
RUN gdown

# input and output folders
RUN mkdir -p ./private/images
RUN mkdir -p ./private/labels
RUN mkdir -p ./output

# !!!! ONLY FOR THE TEST RUN - DELETE BEFORE SUBMITTING --->>>
COPY test_data/images ./private/images
COPY test_data/labels ./private/labels
# <<<---

CMD /bin/sh -c "python3 scripts/solution.py ./weights/weights.py ./config/detection/yolo.py && python3 scripts/scorer.py"

