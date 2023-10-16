# Librairies pour faire les tests avec unittest
import unittest
from unittest.mock import Mock
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
from io import BytesIO
import re

#Initialisation de Flask
app = Flask(__name__)
#app.config.from_object(__name__)
#app.config["SECRET_KEY"] = "74c1112c-d16f-446c-9b6f-ee3315b7ec8b"

#Importation du modèle mlflow
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
    print("Valeur de la référence client : ", ref_client)
    #Création d'un dataframe avec les information du client sélectionné :
    #data_work_client = pd.DataFrame(data_work_complet,index=[ref_client])
    data_work_client = pd.DataFrame(data_work_complet, index=[ref_client])
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
    # On transforme ensuite le dictionnaire en dataframe
    data_work_list_result = pd.DataFrame(data=list_result_work)
    print("Conversion du dictionnaire en dataframe des caractéristiques client : ", data_work_list_result)
    # On oublie pas de préparer la transformation des variables catégorielles en variables numériques plus tard
    transf_data_work_categ = {'Prêts de trésorerie': 0, 'Prêts renouvelables': 1,
                              'autre': 0, 'célibataire': 1, 'marié(e)': 2,
                              'M': 1, 'F': 0,
                              '3 à 4': 0, '5 à 8': 1}
    print("conversion des variables en variables numériques : ", transf_data_work_categ)
    # Découpage des datasets en dataset de train et de test (proportion 80/20)
    X_train, X_test, y_train, y_test = train_test_split(data_work_complet, data_target_complet, test_size=0.2,
                                                        random_state=42)
    # Librairie pour encoder des variables catégorielles
    encoder = LabelEncoder()
    # On remet la variable concernant une période sous un format positif
    X_test['Delai_anticipation_pret'] = X_test['Delai_anticipation_pret'].mul(-1)
    # On remet la variable concernant une période sous un format positif
    X_train['Delai_anticipation_pret'] = X_train['Delai_anticipation_pret'].mul(-1)
    # Création d'une variable avec la liste des colonnes catégorielles du dataset features
    data_categ = list(data_work_complet.select_dtypes(exclude=["number"]).columns)
    print("affichage de data_categ : ", data_categ)
    print("X_train : ", X_train.head())
    print("X_test : ", X_test.head())
    # Encodage des variables catégorielles
    for col in data_categ:
        X_train[col] = encoder.fit_transform(X_train[col])
        X_test[col] = encoder.fit_transform(X_test[col])
    print("X_train_transform : ", X_train.head())
    print("X_test_transform : ", X_test.head())
    # On transforme les variables catégorielles en variables numériques
    data_work_list_result_transf = data_work_list_result.replace(transf_data_work_categ)
    print("Voici le dataset après transformation des variables catégorielles en numériques : ",
          data_work_list_result_transf)
    # Prédiction du résultat
    score = model.predict_proba(data_work_list_result_transf)
    print("affichage du score : ", score)
    return score

#Fonction pour réaliser les tests unitaires
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

#Prédiction du score predictproba
@app.route("/streamlit_predictproba", methods=['POST'])
def streamlit_predictproba():
    print("Lancement de la fonction streamlit_predictproba")
    #Récupération des infos clients
    file = request.files["file"]
    file_read = file.read()
    print("valeur de id_client : ", file_read)
    conv_str = str(file_read)
    print("variable conv_str : ", conv_str)
    res = re.sub(r'[^\d]+', '', conv_str)
    print("variable res : ", res)
    res = int(res)
    score_pred = calc_score_predictproba(res, data_work_complet)
    print("la valeur pred est de : ", score_pred)
    score_pred = round(score_pred[0][0], 3)
    score_list = score_pred.tolist()
    return jsonify({"predict_score": score_list})

#Prédiction de la variable target
@app.route("/streamlit_prediction", methods=['POST'])
def streamlit_prediction():
    print("Lancement de la fonction streamlit_prediction")
    #Récupération des infos clients
    file = request.files["file"]
    file_read = file.read()
    data_file_read = (pd.read_csv(BytesIO(file_read))).T
    print("variable file read : ", data_file_read)
    list_var = data_file_read.iloc[1].tolist()
    print("liste var : ", data_file_read.iloc[1].tolist())
    select_data = pd.DataFrame(list_var).T
    print("Variable select_data : ", select_data)
    #Prédiction du résultat
    pred = model.predict(select_data)
    print("la valeur pred est de : ", pred)
    rec = pred.tolist()
    return jsonify({"predictions": rec})


if __name__ == '__main__':
    print("hello")
    #launch_unittest = unittest.main()
    #print("voici les résultats des tests unitaires : ", launch_unittest)
    app.run(debug=True)
    #app.run(host="0.0.0.0", port=8080)
