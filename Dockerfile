FROM python:3.10
WORKDIR /app
COPY ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt
COPY . .
## Rendre le port 80 accessible au monde extérieur à ce conteneur
EXPOSE 80
## Exécuter app.py lorsque le conteneur est lancé
CMD streamlit run --server.port 80 dashboard.py
