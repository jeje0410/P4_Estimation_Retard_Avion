# P4_Estimation_Retard_Avion
L'objectif est d'estimer le retard des futurs vols d'avion

## Analyse des données
64 dimensions dans le jeu de données
* Quantitatives
* Qualitatives ou catégorielles
* Temporalité

## Préprocessing
* Création de variables
   * Nombre de jours par rapport aux vacances
   * Transformation de la date et l'heure en information cyclique
* One Hot Encoding ou Mean Target Encoding pour les variables catégorielles
* Standardisation des données

## Modélisation
* Création d'un jeu d'entrainement et d'un jeu de test
* Quelles métriques mettre en place pour évaluer mes modèles
   * R2 ou coéfficient de détermination
   * MAE : Moyenne arithmétique des valeurs absolues des écarts
   * MSE :  Moyenne arithmétique des carrés des écars
* Régression linéaire
* Arbre de décision

## API
* Utilisation de Streamlit
