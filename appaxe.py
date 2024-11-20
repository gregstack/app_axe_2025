import tkinter as tk
from tkinter import ttk
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib

# Chemin vers le fichier modèle
model_path = 'trained_model.pkl'
encoder_path = 'label_encoder.pkl'

# Charger le modèle et l'encodeur de labels
model = joblib.load(model_path)
label_encoder = joblib.load(encoder_path)

# Fonction pour préparer les nouvelles données pour la prédiction
def prepare_new_data(masse, nbop, diam, eau, ageop, mat):
    # Encodage de 'MAT'
    mat_encoded = label_encoder.transform([mat])[0]
    
    # Créez un DataFrame avec les nouvelles données
    new_data = pd.DataFrame({
        'MASSE': [masse],
        'NBOP': [nbop],
        'DIAM': [diam],
        'EAU': [eau],
        'AGEOP': [ageop],
        'MAT': [mat_encoded]
    })
    
    return new_data

def predict_price():
    try:
        # Obtenez les valeurs des entrées
        masse = float(entry_masse.get())
        nbop = int(entry_nbop.get())
        diam = float(entry_diam.get())
        eau = float(entry_eau.get())
        ageop = float(entry_ageop.get())
        mat = combo_mat.get()
        
        # Préparation des nouvelles données
        new_data = prepare_new_data(masse, nbop, diam, eau, ageop, mat)
        
        # Prédiction
        predicted_price = model.predict(new_data)
        label_result.config(text=f"Le prix prédit est : {predicted_price[0]:.2f}€")
    except Exception as e:
        label_result.config(text=f"Erreur: {e}")

# Interface Tkinter
root = tk.Tk()
root.title("🚀 Prédiction de prix famille AXES 🎯")

tk.Label(root, text="MASSE (kg):").grid(row=0, column=0)
entry_masse = tk.Entry(root)
entry_masse.grid(row=0, column=1)

tk.Label(root, text="Nombre opérations d'usinage:").grid(row=1, column=0)
entry_nbop = tk.Entry(root)
entry_nbop.grid(row=1, column=1)

tk.Label(root, text="Diamètre (mm):").grid(row=2, column=0)
entry_diam = tk.Entry(root)
entry_diam.grid(row=2, column=1)

tk.Label(root, text="Quantité prévisionnelle:").grid(row=3, column=0)
entry_eau = tk.Entry(root)
entry_eau.grid(row=3, column=1)

tk.Label(root, text="Age opérateur:").grid(row=4, column=0)
entry_ageop = tk.Entry(root)
entry_ageop.grid(row=4, column=1)

tk.Label(root, text="Matière:").grid(row=5, column=0)
combo_mat = ttk.Combobox(root, values=label_encoder.classes_.tolist())
combo_mat.grid(row=5, column=1)

tk.Button(root, text="Prédire le prix ", command=predict_price).grid(row=6, column=0, columnspan=2)

label_result = tk.Label(root, text="")
label_result.grid(row=7, column=0, columnspan=2)

root.mainloop()
