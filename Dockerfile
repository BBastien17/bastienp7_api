#Dockerfile explique comment construire un Docker Image
FROM continuumio/anaconda3:2020.11

ADD . /code
WORKDIR /code

ENTRYPOINT ["python", "app.py"]

