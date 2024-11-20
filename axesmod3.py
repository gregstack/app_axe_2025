import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Chemin du fichier
file_path = '/Users/greg/Workspace/ML toles/axxe1.csv'

# Vérifier si le fichier existe dans le répertoire courant
if not os.path.isfile(file_path):
    print(f"Le fichier {file_path} n'existe pas dans le répertoire actuel.")
    print("Fichiers disponibles dans le répertoire actuel :")
    for file in os.listdir('.'):
        print(file)
else:
    # Charger les données
    data = pd.read_csv(file_path)

    # Définir les colonnes à convertir en numérique
    cols_to_convert = ['MASSE', 'NBOP', 'DIAM', 'EAU', 'AGEOP','PRIX']

    # Convertir uniquement les colonnes spécifiées en numérique avec 'coerce'
    data[cols_to_convert] = data[cols_to_convert].apply(pd.to_numeric, errors='coerce')
    
    # Supprimer les lignes avec des valeurs manquantes
    data.dropna(inplace=True)

    # Encoder les variables catégorielles
    label_encoder = LabelEncoder()
    data['MAT'] = label_encoder.fit_transform(data['MAT'])
    
    # Sauvegarder l'encodeur de labels
    joblib.dump(label_encoder, 'label_encoder.pkl')

    # Nettoyage des données : Détection et suppression des outliers
    Q1 = data[cols_to_convert].quantile(0.25)
    Q3 = data[cols_to_convert].quantile(0.75)
    IQR = Q3 - Q1

    data_cleaned = data[~((data[cols_to_convert] < (Q1 - 1.5 * IQR)) | (data[cols_to_convert] > (Q3 + 1.5 * IQR))).any(axis=1)]
    
    # Séparation des variables X et Y 
    X = data_cleaned[['MASSE','NBOP', 'DIAM','EAU','AGEOP','MAT']]  # Colonnes B à G et MAT encodée
    Y = data_cleaned['PRIX']  # Colonne H

    # Division des données en ensembles d'entraînement et de test
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=0)
    
    # Création et entraînement du modèle
    model = LinearRegression()
    model.fit(X_train, Y_train)

    # Prédictions sur l'ensemble de test
    Y_pred = model.predict(X_test)
    
    # Calcul des métriques d'évaluation : MSE et R^2
    mse = mean_squared_error(Y_test, Y_pred)
    r2 = r2_score(Y_test, Y_pred)

    print(f"Mean Squared Error (MSE) : {mse}")
    print(f"Coefficient de détermination R^2 : {r2}")
    
    # Création de la heatmap pour visualiser la corrélation entre les variables sélectionnées
    selected_columns = ['MASSE','NBOP', 'DIAM', 'EAU','AGEOP','MAT', 'PRIX']
    corr_matrix = data_cleaned[selected_columns].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, vmin=-1, vmax=1)
    plt.title('Heatmap de corrélation entre les variables selectionnées')
    plt.show()

    # Sauvegarder le modèle
    model_path = 'trained_model.pkl'
    joblib.dump(model, model_path)

    print(f'Modèle sauvegardé à : {model_path}')
