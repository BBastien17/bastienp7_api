#Dockerfile explique comment construire un Docker Image
#FROM continuumio/anaconda3:2020.11

#ADD . /code
#WORKDIR /code

#ENTRYPOINT ["python", "app.py"]



FROM     python
WORKDIR  /app
COPY  ./app.py./
COPY	requirements.txt./
RUN		pip install –r requirements.txt --no-cache-dir
CMD		[“gunicorn”,”-w” “4”,”app:app”,”--bind” “0.0.0.0:8000”]
