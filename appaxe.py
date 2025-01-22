from flask import Flask, render_template, request
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Initialisation de l'application Flask
app = Flask(__name__)

# Chemins vers les fichiers du modèle
MODEL_PATH = 'trained_model.pkl'
ENCODER_PATH = 'label_encoder.pkl'

# Charger le modèle et l'encodeur de labels
model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)

# Fonction pour préparer les nouvelles données pour la prédiction
def prepare_new_data(masse, nbop, diam, eau, mat):
    # Encodage de 'MAT'
    mat_encoded = label_encoder.transform([mat])[0]

    # Créez un DataFrame avec les nouvelles données
    new_data = pd.DataFrame({
        'MASSE': [masse],
        'NBOP': [nbop],
        'DIAM': [diam],
        'EAU': [eau],
        'MAT': [mat_encoded]
    })

    return new_data

# Route pour la page d'accueil
@app.route('/')
def index():
    return render_template('index.html', materials=label_encoder.classes_.tolist())

# Route pour gérer la prédiction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Récupérer les données du formulaire
        masse = float(request.form['masse'])
        nbop = int(request.form['nbop'])
        diam = float(request.form['diam'])
        eau = float(request.form['eau'])
        mat = request.form['mat']

        # Préparer les données pour la prédiction
        new_data = prepare_new_data(masse, nbop, diam, eau, mat)

        # Effectuer la prédiction
        predicted_price = model.predict(new_data)

        # Retourner le résultat
        return render_template('result.html', price=predicted_price[0])
    except Exception as e:
        return render_template('result.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
