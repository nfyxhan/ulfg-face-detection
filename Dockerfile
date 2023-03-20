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

ENV TORCH_WHL torch-1.10.1-cp36-cp36m-linux_x86_64.whl 

ADD requirements-cpu.txt requirements.txt
RUN wget https://download.pytorch.org/whl/cpu/torch-1.10.1%2Bcpu-cp36-cp36m-linux_x86_64.whl --no-check-certificate -O ${TORCH_WHL} && \
  pip3 install ${TORCH_WHL} && \
  rm -f ${TORCH_WHL} && \
  pip3 install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --no-cache-dir

ADD vision vision
ADD models models
ADD main.py .
ADD api.py .

CMD python3 ./main.py
