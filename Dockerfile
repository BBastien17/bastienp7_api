FROM python:3.10
WORKDIR /app
COPY ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt
COPY . .
## Rendre le port 80 accessible au monde extérieur à ce conteneur
##EXPOSE 8501
## Exécuter app.py lorsque le conteneur est lancé
##CMD streamlit run dashboard.py --server.port $PORT
CMD ["gunicorn", "app:app"]
