Prédiction de prix pour la famille AXES

Cette application Flask permet de prédire le prix d'un produit basé sur ses caractéristiques techniques. L'utilisateur peut soumettre des données via une interface web simple, et le modèle de machine learning fournit une estimation du prix.

Fonctionnalités

Formulaire interactif pour entrer les caractéristiques du produit.

Prédiction basée sur un modèle pré-entraîné.

Interface utilisateur avec des templates HTML et CSS.

Prérequis

Python 3.8 ou supérieur.

Les bibliothèques suivantes doivent être installées :

pip install flask pandas numpy scikit-learn joblib

Structure du projet

project/
|-- app.py                # Script principal de l'application Flask
|-- trained_model.pkl     # Modèle de machine learning pré-entraîné
|-- label_encoder.pkl     # Encodeur pour les labels
|-- templates/
|   |-- index.html        # Formulaire principal
|   |-- result.html       # Page des résultats
|-- static/
|   |-- styles.css        # Fichier de style CSS
|-- README.md             # Documentation du projet

Exécution

Clonez ce dépôt :

git clone <URL_DU_DEPOT>

Placez les fichiers trained_model.pkl et label_encoder.pkl dans le répertoire racine.

Lancez l'application :

python app.py

Accédez à l'application via http://127.0.0.1:5000.

Personnalisation

Vous pouvez modifier les styles CSS dans le fichier static/styles.css.

Les templates HTML sont situés dans le dossier templates/.

Dépendances

Flask

pandas

numpy

scikit-learn

joblib

Contributeurs

Créé par Grégory et ChatGPT.

Licence

Libre d'utilisation dans un cadre éducatif et non commercial.

