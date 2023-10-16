# bastienp7_api

Explication des fonctionnalités de l'API et du Dashboard réalisés pour l'entreprise Prêt à dépenser.

L’API est développée en Python. J’ai opté pour un hébergement sous Heroku qui permet de relier automatiquement mon projet Github
et de le déployer à chaque nouvelle modification sans avoir à créer l’API à chaque fois. 

L'API représente le backend de notre application par l'intermédiaire du fichier app.py

Elle permet soit de prédire le score de prédiction d'un client ou de prédire la variable Target c'est à dire la variable qui permet de donner un avis
favorable ou défavorable au risque de défaut de paiement d'un client. L'API recoit les données envoyées via des requêtes issues du fichier
dashboard.py qui représente le frontend de notre application.

Le modèle de prédiction est importé via un pickle et il représente le modèle XGBoost généré avec MLFLOW dans les notebooks de préparation

On trouve d’abord la page d’accueil Dashboard.html qui nous permet soit de réaliser une simulation, soit être redirigé vers le Dashboard.

Le fait d’avoir cette interaction entre l’API et le Dashboard est essentiel pour une bonne fluidité lors du rendez-vous avec le client/prospect
et apporter une réponse rapide mais aussi créer une expérience utilisateur de qualité.
