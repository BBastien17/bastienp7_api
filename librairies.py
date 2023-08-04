def list_librairies() :
    import pandas as pd
    import streamlit as st
   # import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import figure
    import requests
    import math
    import seaborn as sns
    from sklearn.preprocessing import LabelEncoder
    import joblib
    from sklearn.metrics import classification_report, confusion_matrix
    #from pycaret.classification import load_model, predict_model
    import mlflow.sklearn
    #import pickle
    #import xgboost
    #from xgboost import XGBClassifier
    #Librairie pour XGBoostClassifier
    from xgboost import XGBClassifier
    import lime
    from lime import lime_tabular
    import shap
    from sklearn.model_selection import train_test_split