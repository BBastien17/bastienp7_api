#Dockerfile explique comment construire un Docker Image
#FROM continuumio/anaconda3:2020.11

#ADD . /code
#WORKDIR /code

#ENTRYPOINT ["python", "app.py"]

FROM python:3.10
WORKDIR /app
COPY ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt
COPY . .
CMD ["gunicorn", "app:app()"]

#FROM python
#WORKDIR /app
#COPY requirements.txt ./

#RUN pip install --no-cache-dir -r requirements.txt
#CMD	[“gunicorn”,”-w” “4”,”app:app”,”--bind” “0.0.0.0:8000”]
