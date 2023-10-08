# Librairies pour faire les tests avec unittest
import unittest
from unittest.mock import Mock
from pathlib import Path
import xgboost.sklearn
from xgboost import sklearn
from xgboost import XGBClassifier
import mlflow.sklearn
import pandas as pd
from flask import Flask, render_template, redirect, request, url_for, send_file, jsonify
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import subprocess
import streamlit as st
from streamlit import runtime
#Librairies pour MLFLOW Tracking
import os
import mlflow
from fastapi import FastAPI
from pydantic import BaseModel
from pandas import json_normalize
import json
import base64


#Importation du modèle mlflow
path = 'Projet_7/'
with open(f'xgb_model_final/model.pkl', 'rb') as f:
  model = pickle.load(f)

#Importation des infos clients
data_work_complet = pd.read_csv("./data_work.csv")
print(data_work_complet.head())
data_target_complet = pd.read_csv("./data_target.csv")
print(data_target_complet.head())

githubAPIURL2 = "https://api.github.com/repos/BBastien17/bastienp7_api/contents/pred.csv"
githubAPIURL3 = "https://api.github.com/repos/BBastien17/bastienp7_api/contents/score.csv"
githubToken = "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"

#Fonction pour calculer le score prédictproba du client
def calc_score_predictproba (ref_client, data_work_complet):
    print("Lancement de la fonction calc_score_predictproba")
    #Création d'un dataframe avec les information du client sélectionné :
    data_work_client = pd.DataFrame(data_work_complet,index=[ref_client])
    print(data_work_client)
    #Création du dictionnaire où l'on stocke les résultats
    list_result_work = {'Type_de_pret':data_work_client['Type_de_pret'], 'Genre':data_work_client['Genre'],
                        'Age':data_work_client['Age'], 'Niveau_d_etudes':data_work_client['Niveau_d_etudes'],
                        'Regime_matrimonial':data_work_client['Regime_matrimonial'],
                        'Nb_enfants': data_work_client['Nb_enfants'],
                        'Nb_membre_famille':data_work_client['Nb_membre_famille'], 
                        'Montant_des_revenus':data_work_client['Montant_des_revenus'],
                        'Note_region_client':data_work_client['Note_region_client'],
                        'Nb_demande_client':data_work_client['Nb_demande_client'],
                        'Montants_du_pret':data_work_client['Montants_du_pret'],
                        'Montant_des_annuites':data_work_client['Montant_des_annuites'],
                        'Nb_jours_credits':data_work_client['Nb_jours_credits'],                       
                        'Montant_anticipation_pret':data_work_client['Montant_anticipation_pret'],
                        'Delai_anticipation_pret':data_work_client['Delai_anticipation_pret']}




def is_sourcefile(path):
    """Retourne True si le fichier est un fichier source Python"""
    if not path.is_file():
        raise Exception("Fichier indisponible")
    return path.suffix == ".py"

class UneClasseDeTest(unittest.TestCase):
    def test_is_sourcefile_when_sourcefile(self):
        path = Mock()
        path.is_file.return_value = True
        path.suffix = ".py"
        resultat = is_sourcefile(path)
        self.assertTrue(resultat)
        path.is_file.assert_called()
    def test_is_sourcefile_when_file_does_not_exist(self):
        path = Mock()
        path.is_file.return_value = False
        with self.assertRaises(Exception):
            is_sourcefile(path)
        path.is_file.assert_called()
    def test_is_sourcefile_when_not_expected_suffix(self):
        path = Mock()
        path.is_file.return_value = True
        path.suffix = ".txt"
        resultat = is_sourcefile(path)
        self.assertFalse(resultat)
        path.is_file.assert_called()
    def test_is_sourcefile_when_not_expected_suffix(self):
        path = Mock()
        path.is_file.return_value = True
        path.suffix = ".pkl"
        resultat = is_sourcefile(path)
        self.assertFalse(resultat)
        path.is_file.assert_called()
    def test_is_sourcefile_when_not_expected_suffix(self):
        path = Mock()
        path.is_file.return_value = True
        path.suffix = ".csv"
        resultat = is_sourcefile(path)
        self.assertFalse(resultat)
        path.is_file.assert_called()
      
app = Flask(__name__)
app.config.from_object(__name__)
app.config["SECRET_KEY"] = "74c1112c-d16f-446c-9b6f-ee3315b7ec8b"
todos = {}

@app.get("/")
def index():     
    return render_template('dashboard.html', todos=todos)

@app.route('/add', methods = ['GET', 'POST'])
def add():
    if request.method == 'POST':
        #todos = {}
        #todos.clear()
        index = len(todos) + 1
        todos[index] = request.form.get("id_client")
        print("Voici la variable todos[index] : ", todos[index])
        print("Voici la variable todos dans la fonction add : ", todos)
        new_todos = todos
        print("Variable new_todos : ", new_todos)
        #Sauver le dictionnaire dans un fichier pickleve dictionary to person_data.pkl file
        with open('dict_data.pkl', 'wb') as fp:
            pickle.dump(new_todos, fp)
        #Permet d'être rediriger vers une autre fonction Python
        #test()
        return redirect(url_for('client_description'))
    #Permet d'être rediriger vers une autre page html
    return render_template('add.html')

@app.route('/client_description', methods = ['GET', 'POST'])
def client_description():
    with open('dict_data.pkl', 'rb') as fp:
        new_todos = pickle.load(fp)
    print("Variable new_todos de client_description : ", new_todos)
    todos = new_todos
    dict_key_select = list(todos)[0]
    print("Voici la variable dict_key_select : ", dict_key_select)
    ref_client = todos[dict_key_select]
    print("Voici la variable ref_client : ", ref_client)
    ref_client = int(ref_client)
    print("Voici ref_client après transformation en int : ", ref_client)
    score_client = calc_score_predictproba(ref_client, data_work_complet)
    print("Voici la variable score_client : ", score_client)
    score_client_accept = round(score_client[0][0], 3)
    print("Voici la variable score_client_accepté : ", score_client_accept)
    if request.method == 'POST':
        return redirect(url_for('index'))
    return render_template('client_description.html', value=ref_client, score=score_client_accept)

@app.route('/dashboard', methods = ['GET', 'POST'])
def dashboard():
    print("Lancement du Dashboard de simulation")
    return (subprocess.run(["python", "./dashboard.py"]))

@app.route("/api/data_stream")
def data_stream():
    csv_url = 'https://raw.githubusercontent.com/BBastien17/bastienp7_api/main/conv_csv_data.csv'
    conv_data_csv = pd.read_csv(csv_url, sep = '\t')
    st.write("variable conv_data_csv : ", conv_data_csv)
    #Prédiction du score pour l'acceptation ou refus du prêt (variable Target)
    pred = model.predict(conv_data_csv)
    #pred = str(pred)
    print("Affichage de la variable target : ", pred)
    pred = pd.DataFrame(pred)
    pred = pred.to_csv(r'pred.csv',sep='\t', index=False)
    with open("pred.csv", "rb") as f:
        # Encoding "my-local-image.jpg" to base64 format
        encodedData = base64.b64encode(f.read())

        headers = {
            "Authorization": f'''Bearer {githubToken}''',
            "Content-type": "application/vnd.github+json"
        }
        data = {
            "message": "Enregistrement du score client target",  # Put your commit message here.
            "content": encodedData.decode("utf-8")
        }

        r = requests.put(githubAPIURL2, headers=headers, json=data)
        print(r.text)  # Printing the response
      
    #Calcul du score client
    score = model.predict_proba(conv_data_csv)
    #score = str(score)
    score = pd.DataFrame(score)
    score = score.to_csv(r'score.csv',sep='\t', index=False)
    print("Affichage du score predictproba : ", score)
    with open("score.csv", "rb") as f:
      # Encoding "my-local-image.jpg" to base64 format
      encodedData = base64.b64encode(f.read())

      headers = {
          "Authorization": f'''Bearer {githubToken}''',
          "Content-type": "application/vnd.github+json"
      }
      data = {
          "message": "Enregistrement du score predictproba",  # Put your commit message here.
          "content": encodedData.decode("utf-8")
      }

      r2 = requests.put(githubAPIURL3, headers=headers, json=data)
      print(r2.text)  # Printing the response
    


if __name__ == '__main__':
    print("hello")
    launch_unittest = unittest.main()
    print("voici les résultats des tests unitaires : ", launch_unittest)
    app.run(debug=False)
    #Pour le fonctionnement en local
    #app.run(debug=True)

