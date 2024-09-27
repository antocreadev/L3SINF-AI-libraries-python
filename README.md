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
Comment traîté la donnée en chaîne de caractère :
Exemple : Chat, chien, oiseau, poisson
il faut transformer les données en données numériques pour les modèles d'apprentissage :
- One hot encoding : chaque catégorie est transformée en une colonne binaire. Il faut faire attention car si on a beaucoup de catégories, on va avoir beaucoup de colonnes. Un modèle d'apprentissage peut être moins précis si on a trop de colonnes et peu de lignes. 
<!-- Pour savoir combien de colonnes on peut avoir, on peut utiliser la formule : 2^(nombre de catégories) - 1 -->
---
Il faut : 
- effacer les données inutiles. (on peut utiliser la corrélation pour voir les données les plus importantes)
- regrouper les données similaires. (on peut utiliser le clustering, utiliser des données des modèles non supervisés pour regrouper les données similaires)
- transformer les données en données numériques (on peut utiliser le one hot encoding pour les données catégorielles ou le label encoding pour les données ordinales)
- normaliser les données (pour avoir une distribution gaussienne)

pour éviter le sur-apprentissage.



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
( pour retrouver le log, il faut faire exp(x) )
c'est important d'avoir une distribution gaussienne pour les modèles d'apprentissage
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

#### K-means : 

- Non supervisé
- Algorithme de clustering
- Partitionne les données en K groupes (clusters) où chaque observation appartient au cluster avec la moyenne la plus proche
- Il est basé sur la minimisation de la variance intra-cluster

l'inertie c'est la distance de chacun des points au centre de gravité total, le nombre de cluseter le plus significatif ce sont ceux dans le coude de la courbe 
K-means n'est pas stable donc on trouve le nombre de cluster avec l'inertie et on utilise cette valeur dans un modèle CAH

#### DBSCAN (Density-Based Spatial Clustering of Applications with Noise) :

- Non supervisé
- Algorithme de clustering basé sur la densité
- Peut trouver des clusters de forme arbitraire
- Peut identifier les points qui ne font partie d'aucun cluster (outliers)

#### Isolation Forest :

- Non supervisé
- Méthode d'identification d'anomalies (outliers)
- Utilise un ensemble d'arbres de décision pour isoler les anomalies dans les données en les considérant comme des points inhabituels

# Cheatsheet
# Pandas 

## Lire un fichier csv avec pandas
```python
pd.read_csv('data.csv')

# pour choisir le séparateur
pd.read_csv('data.csv', sep=';')
```

## Séléctionner des colonnes
1. Séléctionner des colonnes
```python
df[['col1', 'col2']]
```

2. Séléctionner des colonnes avec des conditions
```python
df[df['col1'] == 'condition']
```

3. Séléctionner des colonnes avec plusieurs conditions
```python
df[(df['col1'] == 'condition1') & (df['col2'] == 'condition2')]
```

4. Séléctionner des colonnes par des types
```python
df.select_dtypes(include=['int64'])

df.select_dtypes(include=[np.number])
```

## Séléctionner des lignes
1. `.loc[]` pour sélécionner plusieurs lignes
```python
data.loc[[1,2], "nom_colonne"]

# data.loc permet de récupérer des ligne 
# data.loc[[1,2]] -> récupère les lignes 1 et 2 

# data.loc[[1,2], "nom_colonne"] -> récupère la colonne sex de la ligne 1 et 2
```

2. `.at[]` pour séléctionner une seule ligne
```python
data.at[1, "nom_colonne"]
```

## Récupérer des informations sur le dataframe
1. Récupérer les informations sur le dataframe
```python
df.info()
```
2. Récupérer les statistiques sur le dataframe
```python
df.describe()
```
3. Récupérer les colonnes du dataframe
```python
df.columns
```
4. Récupérer les valeurs uniques d'une colonne
```python
df['colonne'].unique()
```
6. Avoir la taille du dataframe
```python
df.shape
```
7. Récupérer la moyenne d'une colonne
```python
df['colonne'].mean()
```
8. Récupérer la somme d'une colonne
```python
df['colonne'].sum()
```
9. Récupère le type des colonnes
```python
data.dtypes
```

## Récupérer les NaN
```python
df.isna()
```
## Récupérer les lignes avec des NaN
```python
df.isna().any(axis=1)
# axis=1 pour récupérer les lignes
# axis=0 pour récupérer les colonnes
```

## Supprimer des éléments avec un filtre
1. Supprimer les lignes avec des filtres
```python
data.drop(data[filtre].index, inplace=True)

# le premier paramètre est une liste d'index

# inplace=True pour modifier le dataframe
# inplace=False pour ne pas modifier le dataframe
```

2. Supprimer des colonnes
```python
data.drop(columns=['col1', 'col2'], inplace=True)
```

## Compter des valeurs dans une colonne
```python
df['colonne'].value_counts()
```


## Remplacer des valeurs
```python
data["colonne"].replace({1: "Homme", 0: "Femme"}, inplace=True)
```

## Définir une colonne comme index
```python
data.set_index('colonne', inplace=True)
```

## Regarder comment les valeurs se répartissent
```python
pd.cut(data['colonne'], bins=3)
```

## Regarder les corrélations
```python
data.corr()
```

## Trier les données
```python
data.sort_values('colonne', ascending=False)
```

# Sklearn

## Séparer les données
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## Créer un modèle
```python
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cluster import KMeans
...
model = LinearRegression()
```

## Entraîner un modèle
```python
model.fit(X_train, y_train)
```

## Faire des prédictions
```python
model.predict(X_test)
```

## Evaluer un modèle
```python
model.score(X_test, y_test)
```

## Sauvegarder un modèle
```python
import joblib
joblib.dump(model, 'model.pkl')
```

## Charger un modèle
```python
model = joblib.load('model.pkl')
```

## Créer un pipeline
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])
```

## Le PCA (Principal Component Analysis)

c'est une méthode de réduction de dimensionnalité qui permet de réduire le nombre de colonnes d'un dataframe.
permet de savoir quelles colonnes sont les plus importantes / permet de lier les colonnes entre elles
```python
from sklearn.decomposition import PCA

pca = PCA(svd_solver='full')
data = pca.fit_transform(data)
# transforme en DataFrame
data = pd.DataFrame(data)
```

Pour afficher les colonnes les plus importantes
```python
pca.explained_variance_ratio_

# transforme en DataFrame
pd.DataFrame(pca.explained_variance_ratio_)
```
## CAH (Classification Ascendante Hiérarchique)
C'est une méthode de clustering qui permet de regrouper des individus en fonction de leurs caractéristiques.
ça permet de savoir si des individus se ressemblent ou non pour les regrouper ensemble et les différencier des autres. Pour que entrainement soit efficace, il faut que les individus soient proches les uns des autres.
```python
from sklearn.cluster import AgglomerativeClustering
model = AgglomerativeClustering(n_clusters=3)
model.fit(data)

# pour récupérer les labels
model.labels_
```

## CAH avec PCA
```python
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
# model est AgglomerativeClustering et on fit sur les données transformées par PCA
model.fit(dataPCA)
PCA_CAH = pd.DataFrame(model.labels_, index=data.index)
```

## Matrice de confusion
```python
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)
```

# Matplotlib et Seaborn
# Créer un graphique
```python
import matplotlib.pyplot as plt
plt.plot(x, y)
```

# dendrogramme
```python
from scipy.cluster.hierarchy import dendrogram, linkage
Z = linkage(data, 'ward')
dendrogram(Z)
```

# nuage de points
```python
import seaborn as sns
sns.scatterplot(x='col1', y='col2', data=data)
```

# histogramme
```python
sns.histplot(data['col1'])
```
# pairplot
```python
sns.pairplot(data)
```

# scatterplot
```python
sns.scatterplot(x='col1', y='col2', data=data)
```

# Matrice de confusion
```python
import seaborn as sns
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True)
```

