FROM python:3.6-slim-stretch
MAINTAINER marselmustafin
COPY . /sentiment-nn
WORKDIR /sentiment-nn
RUN apt-get update 
RUN apt-get install -y gcc
RUN python3 -m pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.12.0-cp36-cp36m-linux_x86_64.whl \
    && pip3 install -r requirements.txt
CMD python3 ./main.py
