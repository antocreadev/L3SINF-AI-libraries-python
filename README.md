# Outils et techniques de l'intelligence artificielle

## L3 SPI parcours informatique - Université de Corse:

### Différents types d'apprentissages :

- supervisé : relations, sorties désirées, par instruction, gradient ou distance (on connait la sortie et on compare avec la sortie désirée) (reseau de neurones, SVM, KNN, arbre de décision, regression logistique, regression linéaire, regression logistique, naive bayes)
- non-supervisé : structure, par observation, auto-organisation (on ne connait pas la sortie) (K-means, clustering, CAH, GPC, GPR, ACP)
- evolutionnaire : optimum d'une fonction, une fonction, par évolution et sélection, stochastique
- renforcement : lois d'action, récompense, par évolution, différence temporelles

Erreur quadratique moyenne :

$$
E = \frac{1}{2} \sum_{i=1}^{n} (y_i - \hat{y_i})^2
$$

en python :

```python
def mse(y, y_hat):
    return np.sum((y - y_hat)**2) / len(y)
```

On prefere utiliser l'erreur quadratique moyenne car elle est plus simple à dériver et à calculer.
On utilise pas l'erreur absolue car elle n'est pas dérivable en 0 mais au carre oui.

Y -> Continu (Régression) ou Discret (Classification)
Continu : Régression linéaire, Régression polynomiale, Régression logistique
Discret : KNN, Arbre de décision, SVM, Réseau de neurones

# Librairies python

- numpy : permet d'ecrire des fonctions numeriques sur des données composeées exclusivement d'entiers ou de flottants
- pandas : analyse de données, manipule des DataSet de type DataFrame
- matplotlib : visualisation de données sous différents types de graphiques (courbes, histogrammes, nuages de points, etc.)
- seaborn : Data Visualisation basé sur matplotlib, permet de transformer des données en graphiques et diagrammes
- scikit-learn : machine learning, palette de méthodes pour piloter le preprocessing des dataSets, la selection de modèles, l'evaluation des performances, etc.

---

PyTorch :

Avantages :

Flexibilité : PyTorch est connu pour sa flexibilité, ce qui en fait un excellent choix pour la recherche en apprentissage profond. Vous pouvez créer des modèles complexes et définir des graphes de calcul dynamiques plus facilement.
Communauté active : PyTorch bénéficie d'une communauté de recherche et de développement active, ce qui signifie que de nombreuses ressources, tutoriels et bibliothèques complémentaires sont disponibles.
Bon support pour les réseaux récurrents : PyTorch est souvent préféré pour les tâches liées aux réseaux de neurones récurrents (RNN) en raison de sa facilité à gérer les séquences dynamiques.
Débogage facile : Le mode de calcul dynamique de PyTorch facilite le débogage des modèles et la visualisation des valeurs intermédiaires.
Inconvénients :

Moins mature en production : Bien que PyTorch ait progressé dans le déploiement en production, TensorFlow reste généralement préféré pour les déploiements en entreprise en raison de sa maturité dans ce domaine.
Performance : Dans certaines situations, TensorFlow peut être plus performant que PyTorch, en particulier pour l'entraînement de modèles sur de grandes quantités de données grâce à son modèle de calcul statique.
scikit-learn :

Avantages :

Facilité d'utilisation : scikit-learn est convivial, ce qui en fait un excellent choix pour les débutants en apprentissage machine et l'exploration de données.
Large gamme d'algorithmes : Il offre une grande variété d'algorithmes d'apprentissage supervisé et non supervisé pour répondre à diverses tâches.
Documentation complète : scikit-learn possède une documentation détaillée et une communauté d'utilisateurs active, ce qui facilite l'apprentissage et la résolution des problèmes.
Inconvénients :

Principalement pour l'apprentissage machine traditionnel : Il n'est pas conçu pour l'apprentissage profond, donc si vous travaillez principalement avec des réseaux de neurones profonds, d'autres bibliothèques comme TensorFlow ou PyTorch peuvent être nécessaires.
Moins adapté à la recherche en apprentissage profond : Pour les projets de recherche avancée en apprentissage profond, scikit-learn ne propose pas la flexibilité de PyTorch ou TensorFlow.
TensorFlow :

Avantages :

Polyvalence : TensorFlow prend en charge à la fois l'apprentissage machine traditionnel et l'apprentissage profond, ce qui le rend polyvalent pour une grande variété de tâches.
Maturité en production : TensorFlow est largement utilisé dans des applications de production et est souvent préféré pour le déploiement de modèles en entreprise.
TensorFlow Serving : Il offre TensorFlow Serving, un framework de déploiement en production facilitant le déploiement de modèles à grande échelle.
Inconvénients :

Courbe d'apprentissage : TensorFlow peut sembler plus verbeux et difficile à prendre en main pour les débutants en apprentissage machine.
Modèle computationnel statique : La définition d'un graphique de calcul statique peut rendre certaines opérations plus complexes que dans PyTorch.
Documentation moins accessible : Certains utilisateurs trouvent que la documentation de TensorFlow est moins conviviale que celle de PyTorch ou de scikit-learn.

---

Un modèle linéaire si je lui rajoute des données au carré pour améliorier la précision (ou puissance 3,4,5 etc), on peut également lui rajouter des données avec des corrélations pour améliorer la précision.

Cependant cela peut provoquer du sur-apprentissage (overfitting) car le modèle va apprendre les données d'entrainement par coeur et ne sera pas capable de généraliser sur de nouvelles données. Car il apprend du bruit et non des données.

Plus j'apprends de données, plus je risque de faire du sur-apprentissage, donc moins je serais précis.

---

Erreuz de généralisation d'un modèle :

- Erreur de biais : erreur de généralisation d'un modèle, à cause d'une hypothèse trop simpliste : le modèle ne peut pas représenter la relation entre les données et les sorties désirées

- Erreur de variance : erreur de généralisation d'un modèle, à cause d'une hypothèse trop complexe : le modèle représente trop bien la relation entre les données et les sorties désirées, il apprend du bruit et non des données

- Erreur de bruit : erreur de généralisation d'un modèle, à cause d'un bruit dans les données : le modèle ne peut pas représenter la relation entre les données et les sorties désirées

---

un hyper plan sur une dimension X est une courbe "tordu" polymonial en 2D.

---

Le support vector machine est adapté jusqu'à quelques millions de données, au dela il est préférable d'utiliser un réseau de neurones.

réseau de neurones 100 milles données par variable

---

Les modèles d'apprentissages apprennent mieux sur une courbe gaussienne. Car elle est derivable en tout point et donc plus facile à apprendre.

Les modèles d'apprentissage apprennent moins bien sur une courbe en escalier (dirac). Car elle n'est pas derivable en tout point et donc plus difficile à apprendre (il n'y a pas de derivé dans les points de cassures).

On peut utiliser le log pour transformer une courbe en escalier en courbe gaussienne.
(il faut rajouter 1,5 à toutes les valeurs pour éviter d'avoir des valeurs négatives)

---

Linear Regression :

- Supervisé
- Regression
- Trouver la meilleure droite qui passe au milieu des points

SVC (Support Vector Classification) :

- Supervisé
- Classification
- Trouver la meilleure droite qui sépare les points

---

K - Nearest Neighbors (KNN) :

- Supervisé
- Classification
  Calcul la distance entre les points et les classes les plus proches

- distance euclidienne
- distance de manhattan
- distance de minkowski
- distance de hamming
- distance de jaccard

---

Arbre de décision :

- Supervisé
- Classification
- Trouver la meilleure droite qui sépare les points

Entrropie : c'est la mesure de l'incertitude d'une variable aléatoire. Plus l'entropie est élevée, plus l'incertitude est grande. Plus l'entropie est faible, plus l'incertitude est faible.

Abre de décision permet de comprendre les relations entre les variables et les sorties désirées.
On a les informations sur les variables les plus importantes.

---

Les random forest :

- Supervisé
- Classification
- Combinaison de plusieurs arbres de décision

La différence entre les arbres de décision et les random forest est que les random forest sont plus précis car ils combinent plusieurs arbres de décision. (ils prennent des parties de la base de données et font des arbres de décision dessus, puis ils combinent les résultats)

---

Différence entre log et centré/réduire :

- centré/réduire : permet de centrer les données autour de 0 et de les réduire pour avoir une variance de 1 et garde la proportionnalité entre les données
- log : permet de réduire les valeurs élevées et d'augmenter les valeurs faibles pour avoir une distribution gaussienne, ne garde pas la proportionnalité entre les données

---

### Non supervisé

- ACP (Analyse en Composantes Principales)
- CAH (Classification Ascendante Hiérarchique)
- K-means
- DBSCAN
- Isolation Forest

#### ACP (Analyse en Composantes Principales) :

- Non supervisé
- Réduction de dimension
- Trouver les variables les plus importantes


#### CAH (Classification Ascendante Hiérarchique) :

- Non supervisé
- Trouver les groupes de variables les plus proches
- Clustering 

#### K-means 



