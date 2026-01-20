import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Question 9
with open('./data/island-index.csv', encoding= 'utf-8') as file3: 
    contenu = pd.read_csv(file3)
    # Afficher le DataFrame pour vérifier son contenu
    contenu = pd.DataFrame(contenu)
print(contenu)
# Définir les intervalles pour la catégorisation
bins = [0, 10, 25, 50, 100, 2500, 5000, 10000, float('inf')]
labels = ['0-10', '10-25', '25-50', '50-100', '100-2500', '2500-5000', '5000-10000', '≥10000']

# Créer une nouvelle colonne avec les catégories
contenu['Catégorie_Surface'] = pd.cut(contenu['Surface (km²)'], 
                                     bins=bins, 
                                     labels=labels, 
                                     right=True)

# Compter le nombre d'îles dans chaque catégorie
comptage = contenu['Catégorie_Surface'].value_counts().sort_index()

# Afficher les résultats
print("\nNombre d'îles par catégorie de surface :")
print(comptage)

# Créer un graphique à barres pour visualiser la distribution
plt.figure(figsize=(12, 6))
comptage.plot(kind='bar')
plt.title("Distribution des îles par catégorie de surface")
plt.xlabel("Catégorie de surface (km²)")
plt.ylabel("Nombre d'îles")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()