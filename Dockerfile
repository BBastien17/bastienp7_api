FROM python:3.10
WORKDIR /app
COPY ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt
COPY . .
CMD ["sh setup.sh && streamlit", "./dashboard.py --server.port=$PORT"]

