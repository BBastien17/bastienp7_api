# Librairies pour faire les tests avec unittest
import unittest
from unittest.mock import Mock
from pathlib import Path
import xgboost.sklearn
from xgboost import sklearn
from xgboost import XGBClassifier
import mlflow.sklearn
import pandas as pd
from flask import Flask, render_template, redirect, request, url_for, send_file
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

app = FastAPI()

#Importation du modèle mlflow
path = 'Projet_7/'
with open(f'xgb_model_final/model.pkl', 'rb') as f:
  model = pickle.load(f)

#Importation des infos clients
data_work_complet = pd.read_csv("./data_work.csv")
print(data_work_complet.head())
data_target_complet = pd.read_csv("./data_target.csv")
print(data_target_complet.head())

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

#Fonction pour calculer le score prédictproba du client
@app.post("/calc_score_predictproba_streamlit")
def calc_score_predictproba_streamlit (ref_client, data_work_complet):
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
    
    #On transforme ensuite le dictionnaire en dataframe
    data_work_list_result = pd.DataFrame(data=list_result_work)
    print("Conversion du dictionnaire en dataframe des caractéristiques client : ", data_work_list_result)
    #On oublie pas de préparer la transformation des variables catégorielles en variables numériques plus tard
    transf_data_work_categ = {'Prêts de trésorerie': 0, 'Prêts renouvelables': 1,
                              'autre': 0, 'célibataire': 1, 'marié(e)': 2,
                              'M': 1, 'F': 0,
                              '3 à 4': 0, '5 à 8': 1}    
    print("conversion des variables en variables numériques : ", transf_data_work_categ)
    #Découpage des datasets en dataset de train et de test (proportion 80/20)
    X_train, X_test, y_train, y_test = train_test_split(data_work_complet, data_target_complet, test_size = 0.2, random_state=42)
    #Librairie pour encoder des variables catégorielles
    encoder = LabelEncoder()
    #On remet la variable concernant une période sous un format positif
    X_test['Delai_anticipation_pret'] = X_test['Delai_anticipation_pret'].mul(-1)
    #On remet la variable concernant une période sous un format positif
    X_train['Delai_anticipation_pret'] = X_train['Delai_anticipation_pret'].mul(-1)
    #Création d'une variable avec la liste des colonnes catégorielles du dataset features
    data_categ = list(data_work_complet.select_dtypes(exclude=["number"]).columns)
    print("affichage de data_categ : ", data_categ)
    print("X_train : ", X_train.head())
    print("X_test : ", X_test.head())
    #Encodage des variables catégorielles
    for col in data_categ:
        X_train[col] = encoder.fit_transform(X_train[col])
        X_test[col] = encoder.fit_transform(X_test[col])
    print("X_train_transform : ", X_train.head())
    print("X_test_transform : ", X_test.head())
    #On transforme les variables catégorielles en variables numériques
    data_work_list_result_transf = data_work_list_result.replace(transf_data_work_categ)
    print("Voici le dataset après transformation des variables catégorielles en numériques : ", data_work_list_result_transf)
    #Prédiction du résultat
    score = model.predict_proba(data_work_list_result_transf)
    print("affichage du score : ", score)
    return score

#app = FastAPI()



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
#def dashboard():
#    print("Lancement du Dashboard de simulation")
#    subprocess.run(["python", "./dashboard.py"])

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

#Création d'une classe pour les valeurs du client
class User_input(BaseModel):
    Type_de_pret : int
    Genre : int
    Age : int
    Niveau_d_etudes : int
    Regime_matrimonial : int
    Nb_enfants : int
    Nb_membre_famille : int
    Montant_des_revenus : int
    Note_region_client : int
    Nb_demande_client : int
    Montants_du_pret : int
    Montant_des_annuites : int
    Nb_jours_credits : int
    Montant_anticipation_pret : int
    Delai_anticipation_pret : int

def prediction_streamlit(Type_de_pret, Genre, Age, Niveau_d_etudes,
                         Regime_matrimonial, Nb_enfants, Nb_membre_famille,
                         Montant_des_revenus, Note_region_client,
                         Nb_demande_client, Montants_du_pret,
                         Montant_des_annuites, Nb_jours_credits,
                         Montant_anticipation_pret, Delai_anticipation_pret) :
    #dict = {data_list_json.Type_de_pret, data_list_json.Genre,
    #        data_list_json.Age, data_list_json.Niveau_d_etudes,
    #        data_list_json.Regime_matrimonial, data_list_json.Nb_enfants,
    #        data_list_json.Nb_membre_famille, data_list_json.Montant_des_revenus,
    #        data_list_json.Note_region_client, data_list_json.Nb_demande_client,
    #        data_list_json.Montants_du_pret, data_list_json.Montant_des_annuites,
    #        data_list_json.Nb_jours_credits, data_list_json.Delai_anticipation_pret,
    #        data_list_json.Delai_anticipation_pret}
    Nb_enfants = Nb_enfants                       
    dict = {"Nb_enfants":Nb_enfants}
    print("dict dans la fonction prediction_streamlit : ", dict)
    return dict


@app.route("/streamlit_prediction")
def streamlit_prediction():#input:User_input):
    selector = request.args.post("data_list")
    print("variable selector : ", selector)

    dict= json.loads(selector)
    print("variable dict : ", dict)
    data_stream = json_normalize(dict[Type_de_pret, Genre, Age, Niveau_d_etudes,
                                 Regime_matrimonial, Nb_enfants, Nb_membre_famille,
                                 Montant_des_revenus, Note_region_client,
                                 Nb_demande_client, Montants_du_pret,
                                 Montant_des_annuites, Nb_jours_credits,
                                 Montant_anticipation_pret, Delai_anticipation_pret]) 
  
    #data_stream = pd.DataFrame(data=selector)
    print("variable data_stream : ", data_stream)
    #result_dict = prediction_streamlit(input.Type_de_pret, input.Genre,
    #                                   input.Age, input.Niveau_d_etudes,
    #                                   input.Regime_matrimonial, input.Nb_enfants,
    #                                   input.Nb_membre_famille, input.Montant_des_revenus,
    #                                   input.Note_region_client, input.Nb_demande_client,
    #                                   input.Montants_du_pret, input.Montant_des_annuites,
    #                                   input.Nb_jours_credits, input.Delai_anticipation_pret,
    #                                   input.Delai_anticipation_pret)
    #print("variable result_dict : ", result_dict)
    #dict= json.loads(data_list_json)
    #print("variable dict : ", dict)
    #data_list_result_transf = pd.DataFrame.from_dict(dict)
    #data_list_result_transf = data_list_json
    #print("variable data_list_result_transf : ", data_list_result_transf)
    #Prédiction du résultat
    #pred = model.predict(data_list_result_transf)
    #Utile pour les tests
    #st.text(pred)
    #return pred


if __name__ == '__main__':
    print("hello")
    launch_unittest = unittest.main()
    print("voici les résultats des tests unitaires : ", launch_unittest)
    app.run(debug=False)
    #Pour le fonctionnement en local
    #app.run(debug=True)

