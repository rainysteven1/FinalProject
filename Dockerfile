FROM python:3.10-slim

WORKDIR /app

COPY ./requirements.txt /app

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install -r requirements.txt
RUN python -m spacy download en_core_web_sm
