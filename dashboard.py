#Importation des librairies
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import requests

#Importation des fonctions contenants les pages
from page_prospect import page_p
from page_client import page_c

print("Ouverture de la page dashboard.py")

def request_prediction(model_uri, data):
    headers = {"Content-Type": "application/json"}

    data_json = {'data': data}
    response = requests.request(
        method='POST', headers=headers, url=model_uri, json=data_json)

    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))

    return response.json()

#def main():

#Création d'un side bar pour choisir entre une page pour les clients et les prospects
page = st.sidebar.selectbox('Page Navigation', ["Prospects", "Clients"])
st.sidebar.markdown("""---""")
st.sidebar.write("Created by Bastien B")
st.sidebar.image("./images/logo2.png", width=100)

path = 'Projet_7/'

#Import des données clients
data_work = pd.read_csv("./data_work.csv")
data_target = pd.read_csv("./data_target.csv")
data_complete = data_work.copy()
data_complete["Target"] = data_target

#Appel de la page correspondante en fonction du choix réalisé dans le menu déroulant
if page == "Prospects": 
    page_p(data_work, data_target, data_complete)
    
if page == "Clients":
    page_c(data_work, data_target, data_complete)
    
