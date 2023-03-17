FROM  ubuntu:18.04
ENV  LANGUAGE zh_CN.UTF-8
ENV  LANG zh_CN.UTF-8
ENV  LC_ALL zh_CN.UTF-8
RUN apt-get update && apt-get install -y \
  git curl vim wget cmake \
  locales \
  python3 python3-pip \
  libsm6 libxext6 libxrender-dev && \
  locale-gen zh_CN && \
  locale-gen zh_CN.UTF-8 && \
  pip3 install -U pip setuptools

WORKDIR /workdir

ADD requirements.txt .

RUN pip3 install -r requirements.txt

ADD vision vision
ADD models models
ADD detect_imgs.py .
