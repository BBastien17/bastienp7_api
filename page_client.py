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
from xgboost import XGBClassifier
import shap
import lime
from lime import lime_tabular
from sklearn.model_selection import train_test_split
import streamlit.components.v1 as components
#import pickle
import pickle5 as pickle

#Importation du modèle mlflow
path = 'Projet_7/'
model = mlflow.sklearn.load_model('xgb_model_final/')
#Importation des pickles enregistrés
#load_fichierSauvegarde = open("shap_model_X_train", "rb")
#shap_values_1 = pickle.load(load_fichierSauvegarde)
#load_fichierSauvegarde.close()
#load_fichierSauvegarde2 = open("shap_model_data_X_train", "rb")
#explainer = pickle.load(load_fichierSauvegarde2)
#load_fichierSauvegarde2.close()
#commenter les 3 lignes suivantes pour les tests
#load_fichierSauvegarde3 = open("shap_model_data_X_test","rb")
#shap_values_2 = pickle.load(load_fichierSauvegarde3)
#load_fichierSauvegarde3.close()

#Utilisation d'un fonction pour comparer un individu au reste de la population
def compare_client(data_work, data_client):
    data_num = data_work.select_dtypes(['int64','float64'])
    data_num_client = data_client.select_dtypes(['int64','float64'])
    i = 0
    for data_work in data_num:
        #Première ligne pour vérifier la valeur de l'individu
        #st.write(data_num_client.iloc[0, i])
        fig = plt.figure(figsize=(20, 5))
        sns.boxplot(x=data_num[data_work])
        plt.axvline(data_num_client.iloc[0, i], color='red', label='Individu', linewidth=4)
        st.pyplot(fig)
        i = i + 1


#Utilisation d'une fonction pour définir la page client
def page_c (data_work, data_target, data_complete) :
    st.title("Demande d'étude de financement")

    #On rentre l'index du client en lien avec le dataset général
    id_client = st.number_input("Référence du client (Index)",
                                min_value=0, value=0, max_value=307511, step=1)

    #Sélection du client en utilisant un bouton de validation
    valid_btn = st.button('Résultat de la demande de financement', key = "client_button")
    if valid_btn:

        #Création d'un dataframe avec les information du client sélectionné :
        data_client = pd.DataFrame(data_work,index=[id_client])
        data_target_client = pd.DataFrame(data_target,index=[id_client])
        target = data_target_client['TARGET'].values

        #Création du dictionnaire où l'on stocke les résultats
        list_result = {'Type_de_pret':data_client['Type_de_pret'], 'Genre':data_client['Genre'],
                       'Age':data_client['Age'], 'Niveau_d_etudes':data_client['Niveau_d_etudes'],
                       'Regime_matrimonial':data_client['Regime_matrimonial'],
                       'Nb_enfants': data_client['Nb_enfants'],
                       'Nb_membre_famille':data_client['Nb_membre_famille'], 
                       'Montant_des_revenus':data_client['Montant_des_revenus'],
                       'Note_region_client':data_client['Note_region_client'],
                       'Nb_demande_client':data_client['Nb_demande_client'],
                       'Montants_du_pret':data_client['Montants_du_pret'],
                       'Montant_des_annuites':data_client['Montant_des_annuites'],
                       'Nb_jours_credits':data_client['Nb_jours_credits'],                       
                       'Montant_anticipation_pret':data_client['Montant_anticipation_pret'],
                       'Delai_anticipation_pret':data_client['Delai_anticipation_pret']}
    
        #On transforme ensuite le dictionnaire en dataframe
        data_list_result = pd.DataFrame(data=list_result)
        #On oublie pas de préparer la transformation des variables catégorielles en variables numériques plus tard
        transf_data_categ = {'Prêts de trésorerie': 0, 'Prêts renouvelables': 1,
                             'autre': 0, 'célibataire': 1, 'marié(e)': 2,
                             'M': 1, 'F': 0,
                             '3 à 4': 0, '5 à 8': 1}    


        #On affiche les données du client
        st.write('Type de prêt :  \n', data_list_result.iat[0,0])
        st.write('  \n', 'Genre :  \n', data_list_result.iat[0,1])
        st.write('  \n', 'Âge du client :  \n', data_list_result.iat[0,2])
        st.write('  \n', "Niveau d'études :  \n", data_list_result.iat[0,3])
        st.write('  \n', 'Régime matrimoniale :  \n', data_list_result.iat[0,4])
        st.write('  \n', "Nombre d'enfants :  \n", data_list_result.iat[0,5])
        st.write('  \n', "Nombre de membres de la famille :  \n", data_list_result.iat[0,6])
        st.write('  \n', 'Montant annuel des revenus :  \n', data_list_result.iat[0,7])
        st.write('  \n', "Note d'évaluation du lieu d'habitation :  \n", data_list_result.iat[0,8])
        st.write('  \n', 'Nombre de demande de financement du client :  \n', data_list_result.iat[0,9])
        st.write('  \n', 'Montant du prêt :  \n', data_list_result.iat[0,10])
        st.write('  \n', 'Montant des annuités :  \n', data_list_result.iat[0,11])
        st.write('  \n', 'Nombre de jour de crédit :  \n', data_list_result.iat[0,12])
        st.write('  \n', "Délai d'anticipation :  \n", data_list_result.iat[0,13])
        st.write('  \n', "Montant de l'anticipation :  \n", data_list_result.iat[0,14], "  \n")   

        #On affiche le résultat de l'étude
        if target == 0 :
            st.text("Les données fournies permettent d'émettre un avis favorable à la demande de prêt.")
            st.text("Positionnement des caractéristiques du clients vis à vis du reste de la clientèle :")
            compare_client(data_work, data_client)

        elif target == 1 :
            st.text("Les données fournies ne permettent pas d'émettre un avis favorable à la demande de prêt.")

            #Utilisation de  et affichage de l'interprétabilité locale
            #fig = shap.plots.bar(shap_values_2[id_client])
            plt.savefig('shap_report_P7.png', bbox_inches='tight')
            st.image('shap_report_P7.png')
            st.text("Positionnement des caractéristiques du clients vis à vis du reste de la clientèle :")
            compare_client(data_work, data_client)

        else :
            st.text("Error")