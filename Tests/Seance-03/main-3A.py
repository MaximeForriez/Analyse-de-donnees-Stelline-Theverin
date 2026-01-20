import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Source des données : https://www.data.gouv.fr/datasets/election-presidentielle-des-10-et-24-avril-2022-resultats-definitifs-du-1er-tour/

# Sources des données : production de M. Forriez, 2016-2023

# Question 4
with open('./data/resultats-elections-presidentielles-2022-1er-tour.csv') as file2:
    contenu = pd.read_csv(file2)
# Afficher le DataFrame pour vérifier son contenu
contenu = pd.DataFrame(contenu)
print(contenu)

# Question 5
# Copie du code de la Séance 2
# Afficher le type de chaque colonne
print("\nTypes de chaque colonne:")
print(contenu.dtypes)
# Afficher le nom des colonnes 
print("\nNoms des colonnes:")
print(contenu.columns)
# Liste pour stocker la somme de chaque colonne
somme_colonnes = []
# Boucle sur chaque colonne
for colonne in contenu.columns:
    # Calculer la somme si la colonne est quantitative
    if contenu[colonne].dtype in ['int64', 'float64']:
        somme = contenu[colonne].sum()
        somme_colonnes.append((colonne, somme))
# Afficher la liste des sommes des colonnes quantitatives
print("\nSomme des colonnes quantitatives:")
for colonne, somme in somme_colonnes:
    print(f"{colonne}: {somme}")


# Question 5.1
# Sélectionner uniquement les colonnes numériques
colonnes_quantitatives = contenu.select_dtypes(include=['int64', 'float64'])
# Calculer la moyenne de chaque colonne quantitative
moyennes = colonnes_quantitatives.mean()
# Convertir les moyennes en DataFrame et afficher
moyennes_df = moyennes.to_frame(name='Moyenne')
print("\nMoyennes des colonnes quantitatives:")
print(moyennes_df.round(2))

# Question 5.2
# Sélectionner uniquement les colonnes numériques
colonnes_quantitatives = contenu.select_dtypes(include=['int64', 'float64'])
# Calculer la médiane de chaque colonne quantitative
medians = colonnes_quantitatives.median()
# Convertir les médianes en DataFrame et afficher
medians_df = medians.to_frame(name='Médiane')
print("\nMédianes des colonnes quantitatives:")
print(medians_df)

# Question 5.3
# Sélectionner uniquement les colonnes numériques
colonnes_quantitatives = contenu.select_dtypes(include=['int64', 'float64'])
# Calculer les modes de chaque colonne quantitative
modes = colonnes_quantitatives.mode().iloc[0]
# Convertir les modes en DataFrame et afficher
modes_df = modes.to_frame(name='Mode')
print("\nModes des colonnes quantitatives:")
print(modes_df)

# Question 5.4
# Sélectionner uniquement les colonnes numériques
colonnes_quantitatives = contenu.select_dtypes(include=['int64', 'float64'])
# Calculer l'écart-type de chaque colonne quantitative
ecarts_types = colonnes_quantitatives.std()
# Convertir les écarts-types en DataFrame et afficher
ecarts_types_df = ecarts_types.to_frame(name="Écart-type")
print("\nÉcarts-types des colonnes quantitatives:")
print(ecarts_types_df.round(2))

# Question 5.5
# Sélectionner uniquement les colonnes numériques
colonnes_quantitatives = contenu.select_dtypes(include=['int64', 'float64'])
# Calculer l'écart absolu à la moyenne de chaque colonne quantitative
ecarts_absolus = colonnes_quantitatives.apply(lambda x: np.abs(x - x.mean()))
# Afficher les écarts absolus
print("\nÉcarts absolus à la moyenne des colonnes quantitatives:")
print(ecarts_absolus.round(2))

# Question 5.6
# Sélectionner uniquement les colonnes numériques
colonnes_quantitatives = contenu.select_dtypes(include=['int64', 'float64'])
# Calculer l'étendue de chaque colonne quantitative
etendues = colonnes_quantitatives.max() - colonnes_quantitatives.min()
# Convertir les étendues en DataFrame et afficher
etendues_df = etendues.to_frame(name="Étendue")
print("\nÉtendues des colonnes quantitatives:")
print(etendues_df)

# Question 6
# Créer un DataFrame combiné avec tous les paramètres
parametres_combines = pd.concat([
    moyennes_df,
    medians_df,
    modes_df,
    ecarts_types_df,
    etendues_df
], axis=1)
# Afficher le tableau récapitulatif
print("\nRécapitulatif des paramètres statistiques:")
print(parametres_combines.round(2))

# Question 7
# Calculer la distance interquartile (IQR) pour chaque colonne quantitative
colonnes_quantitatives = contenu.select_dtypes(include=['int64', 'float64'])
Q1 = colonnes_quantitatives.quantile(0.25)
Q3 = colonnes_quantitatives.quantile(0.75)
IQR = Q3 - Q1
# Afficher l'IQR
print("\nDistance interquartile (IQR) des colonnes quantitatives:")
for colonne, iqr in IQR.items():
    print(f"{colonne}: {iqr}")
# Calculer la distance interdecile (IDR) pour chaque colonne quantitative
D1 = colonnes_quantitatives.quantile(0.1)
D9 = colonnes_quantitatives.quantile(0.9)
IDR = D9 - D1
# Afficher l'IDR
print("\nDistance interdecile (IDR) des colonnes quantitatives:")
for colonne, idr in IDR.items():
    print(f"{colonne}: {idr}")

# Question 8
# Création du dossier img
import os
if not os.path.exists('img'):
    os.makedirs('img')
# Sélectionner les colonnes quantitatives
colonnes_quantitatives = contenu.select_dtypes(include=['int64', 'float64'])
# Créer une boîte à moustaches pour chaque colonne
for colonne in colonnes_quantitatives.columns:
    # Créer une nouvelle figure pour chaque graphique
    plt.figure(figsize=(10, 6))
    # Créer la boîte à moustaches
    plt.boxplot(colonnes_quantitatives[colonne], labels=[colonne])
    # Ajouter un titre
    plt.title(f'Boîte à moustaches - {colonne}')
    # Rotation des labels si nécessaire
    plt.xticks(rotation=45)
    # Ajuster la mise en page
    plt.tight_layout()
    # Sauvegarder le graphique
    plt.savefig(f'img/boxplot_{colonne}.png')
    # Fermer la figure pour libérer la mémoire
    plt.close()

print("Les boîtes à moustaches ont été sauvegardées dans le dossier 'img'")
# Pour afficher toutes les boîtes à moustaches
fig, axes = plt.subplots(nrows=len(colonnes_quantitatives.columns), figsize=(10, 6*len(colonnes_quantitatives.columns)))
for i, colonne in enumerate(colonnes_quantitatives.columns):
    axes[i].boxplot(colonnes_quantitatives[colonne])
    axes[i].set_title(f'Boîte à moustaches - {colonne}')
plt.tight_layout()
plt.show()