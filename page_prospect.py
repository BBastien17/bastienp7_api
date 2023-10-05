#Importation des librairies
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import figure
import requests
import math
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import mlflow.sklearn
#Librairie pour XGBoostClassifier
from xgboost import XGBClassifier
import shap
from sklearn.model_selection import train_test_split
import pickle


#Importation du modèle mlflow
path = 'Projet_7/'
with open(f'xgb_model_final/model.pkl', 'rb') as f:
    model = pickle.load(f)
#Importation des pickles enregistrés
#load_fichierSauvegarde = open("shap_model_X_train", "rb")
#shap_values_1 = pickle.load(load_fichierSauvegarde)
#load_fichierSauvegarde.close()

#commenter les 3 lignes suivantes pour les tests
#load_fichierSauvegarde2 = open("shap_model_data_X_train", "rb")
#explainer = pickle.load(load_fichierSauvegarde2)
#load_fichierSauvegarde2.close()


#load_fichierSauvegarde3 = open("shap_model_data_X_test","rb")
#shap_values_2 = pickle.load(load_fichierSauvegarde3)
#load_fichierSauvegarde3.close()

#Utilisation d'un fonction pour comparer un individu au reste de la population
def compare_client(data_work, data_list_result):
    data_num = data_work.select_dtypes(['int64','float64'])
    data_num_prospect = data_list_result.select_dtypes(['int64','float64'])
    i = 0
    for data_work in data_num:
        #Première ligne pour vérifier la valeur de l'individu
        #st.write(data_num_client.iloc[0, i])
        fig = plt.figure(figsize=(20, 5))
        sns.boxplot(x=data_num[data_work])
        plt.axvline(data_num_prospect.iloc[0, i], color='red', label='Individu', linewidth=4)
        st.pyplot(fig)
        i = i + 1

#Utilisation d'une fonction pour définir la page prospect
#Création d'un formulaire à compléter
def page_p (data_work, data_target, data_complete) :

    st.title("Demande d'étude de financement :")

    type_pret_client = st.radio("Type de prêt :",
                                key="Type_de_pret",
                                options=['Prêts de trésorerie', 'Prêts renouvelables'])
    
    genre_client = st.radio("Genre :",
                            key="Genre",
                            options=['Masculin', 'Féminin'])
    
    age_client = st.number_input('Âge du client',
                                 min_value=18, value=25, max_value=100, step=1)
    
    niveau_etudes_client = st.radio("Niveau d'études :",
                                    key="Niveau_d_etudes",
                                    options=['3 à 4', '5 à 8'])
    
    reg_matrimonial = st.radio("Régime matrimoniale :",
                               key="Regime_matrimonial",
                               options=['célibataire', "marié(e)", 'autre'])
    
    nb_enfant = st.number_input("Nombre d'enfants",
                                min_value=0, value=0, max_value=19, step=1)
    
    nb_membre_famille = st.number_input("Nombre de membres de la famille",
                                        min_value=1, value=1, max_value=20, step=1)

    mt_revenus = st.number_input('Montant annuel des revenus',
                                 min_value=0, value=40000, max_value=117000000, step=1)
    
    lieu_habitation = st.number_input("Note d'évaluation du lieu d'habitation",
                                      min_value=1, value=1, max_value=3, step=1)
    
    nb_demande = st.number_input('Nombre de demande de financement du client',
                                 min_value=0, value=0, max_value=25, step=1)

    mt_prets = st.number_input('Montant des prêts',
                               min_value=0, value=0, max_value=4050000, step=1)

    mt_annuites = st.number_input('Montant des annuités',
                                  min_value=0, value=0, max_value=300425, step=1)

    nb_jours_credits = st.number_input('Nombre de jours de crédits',
                                       min_value=0, value=0, max_value=2922, step=1)
    
    mt_anticipation = st.number_input("Montant de l'anticipation",
                                      min_value=-169033, value=0, max_value=555000, step=1)

    delai_anticipation = st.number_input("Délai d'anticipation",
                                         min_value=-605, value=0, max_value=392, step=1)
        

    #Création d'un dictionnaire où l'on stocke les résultats
    list_result = {'Type_de_pret':[type_pret_client], 'Genre':[genre_client],
                   'Age':[age_client], 'Niveau_d_etudes':[niveau_etudes_client],
                   'Regime_matrimonial':[reg_matrimonial], 'Nb_enfants': [nb_enfant],
                   'Nb_membre_famille':[nb_membre_famille], 'Montant_des_revenus':[mt_revenus],
                   'Note_region_client':[lieu_habitation], 'Nb_demande_client':[nb_demande],
                   'Montants_du_pret':[mt_prets], 'Montant_des_annuites':[mt_annuites],
                   'Nb_jours_credits':[nb_jours_credits], 'Montant_anticipation_pret':[mt_anticipation],
                   'Delai_anticipation_pret':[delai_anticipation]}

    #On transforme ensuite le dictionnaire en dataframe
    data_list_result = pd.DataFrame(data=list_result)
    #On oublie pas de préparer la transformation des variables catégorielles en variables numériques plus tard
    transf_data_categ = {'Prêts de trésorerie': 0, 'Prêts renouvelables': 1,
                         'autre': 0, 'célibataire': 1, 'marié(e)': 2,
                         'Masculin': 1, 'Féminin': 0,
                         '3 à 4': 0, '5 à 8': 1}   

    #Utile pour les tests afin de vérifiers que nous avons que des variables numériques  
    #list_of_data_type = data_pred.info()
    #st.write(list_of_data_type)
    #st.text(data_pred.info(verbose=True))
     
    #Création d'un bouton pour lancer le scoring
    predict_btn = st.button('Résultat de la demande de financement', key = "prospects_button")
    if predict_btn:
        st.write("predict button was pressed")
        #res = requests.post(url = "http://127.0.0.1:8000/streamlit_prediction")
        

        
