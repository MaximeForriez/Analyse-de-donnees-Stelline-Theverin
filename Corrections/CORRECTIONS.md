# Élements de corrections

## Séance 2.

### Questions

- **Question 6.** Il manque l'explication sur le choix de la représentation.

### Code

- **Question 11.** Vous y étiez presque avec `plt.savefig('diagramme_{dept}.png')`. Il suffisait d'écrire `plt.savefig(f"diagramme_{dept}.png")` pour que la boucle fonctionne.

- **Questions 12 à 13.** non répondues.

## Séance 3.

### Questions

### Code

- Excellent !

## Séance 4

### Questions

- Excellent !

### Code

- Excellent !

## Séance 5

### Questions

- Excellent !

### Code

- Excellent !

- La distribution test1 est normale. Il y a un problème avec le calcul de votre *p-value*. C'est effectivement un faux négatif. Bravo !

## Séance 6

### Questions

- Excellent !

### Code

- **Question 7.** On ne peut pas faire un test sur un seul classement. Il en faut au moins deux.

- Les tests sont faux. Vous avez écrire :

```
    spearman_res = stats.spearmanr(ranks_pop, ranks_dens)
    kendall_res = stats.kendalltau(ranks_pop, ranks_dens)
```

Il aurait fallu écrire :

```
    spearman_res_pop = stats.spearmanr(ranks_pop_2007, ranks_pop_2025)
    kendall_res_pop = stats.kendalltau(ranks_pop_2007, ranks_pop_2025)

    spearman_res_dens = stats.spearmanr(ranks_dens_2007, ranks_dens_2025)
    kendall_res_dens = stats.kendalltau(ranks_dens_2007, ranks_dens_2025)
```

## Humanités numériques

- Aucune analyse rendue.

## Remarques générales

- Aucun dépôt régulier sur `GitHub`.

- Fichiers de code non rendus dans le bon format. J'ai autre chose à faire qu'à convertir vos fichiers. Néanmoins, je ne vous ai pas enlevé de points, car leur conversion, bien que prenant du temps, a été plutôt rapide.

- Attention ! Il ne faut jamais utiliser les adresses absolues `'c:/Users/stell/Downloads/resultats-elections-presidentielles-2022-1er-tour.csv'`, mais les adresses relatives `'./data/resultats-elections-presidentielles-2022-1er-tour.csv'`.

- Concernant l'organisation du cours, je vous rappelle qu'il y avait trois parcours. Qu'aucun étudiant ne m'a jamais précisé le choix de son niveau, même lorsque je l'ai demandé, et que si vous vouliez des précisions, il suffisait de me poser des questions en début de séance ou sur le `Discord`. Si personne ne pose des questions, c'est que tout va bien. L'autonomie ne signifie pas l'indépendance.

- Il est vrai que la plateforme `Jupyter` est très pratique. Je ne vous l'ai pas fait utiliser parce que la plateforme peut devenir payante à tout moment. C'est arrivé avec la plateforme précédente que j'utilise `Replit`. Après dix connexions, ce qui est peu, elle devient payante.

- Excellent travail !
