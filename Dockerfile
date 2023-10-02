#Dockerfile explique comment construire un Docker Image
FROM continuumio/anaconda3:2020.11

#ADD . /code
#WORKDIR /code

#ENTRYPOINT ["python", "app.py"]


WORKDIR /app

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

#ENV GUNICORN_CMD_ARGS="--bind=0.0.0.0 --chdir=./src/"
#COPY . .

#EXPOSE 8000

CMD [ "gunicorn", "app:app" ]

