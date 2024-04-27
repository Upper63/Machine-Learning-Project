#!/usr/bin/env python
# coding: utf-8

# # Projet de Machine Learning

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import pandas as pd
from seaborn import heatmap

import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LinearRegression
get_ipython().run_line_magic('matplotlib', 'inline')


# # Lecture des données

# In[4]:


house = pd.read_csv("kc_house_data.csv",
                    sep=",",
                    decimal='.',
                    encoding="utf-8",
                    header='infer')


# Cet ensemble de données contient **les prix de vente des maisons pour le comté de King**, qui comprend Seattle. Il comprend les maisons vendues entre mai 2014 et mai 2015.

# Il contient **21 variables** : 
# 
# - id : Identifiant d'une maison 
# - date : Date de vente d'une maison
# - price : Le prix de vente de la maison
# - bedrooms : Nombre de chambres
# - bathrooms : Nombre de salles de bain
# - sqft_living : Superficie de la maison
# - sqft_lot : Superficie du terrain
# - floors : Nombre total d'étages (niveaux) dans la maison
# - waterfront : Maison avec vue sur un front de mer
# - view : A été vu (ou visité)
# - condition : L'état général de la maison
# - grade :  Note globale attribuée à l'unité de logement
# - sqft_above : Superficie de la maison sans le sous-sol
# - sqft_basement: Superficie du sous-sol
# - yr_built : Année de construction
# - yr_renovated :Année de rénovation de la maison
# - zipcode : Code postal
# - lat : Coordonnées de latitude de la maison
# - long : Coordonnée de longitude de la maison 
# - sqft_living15 : Surface du salon en 2015 
# - sqft_lot15 : Superficie du terrain en 2015
# 
# Il y a eu **21 613 maisons vendues entre mai 2014 et mai 2015** ce qui correspond au **nombres de lignes de notre data frame.**

# In[3]:


# localisation des maisons de notre dataset à vendre.

import folium
latitude= house['lat'].mean() # latitude moyenne pour représenter la latitude des maisons de notre dataset

longitude = house['long'].mean() #longitude moyenne pour représenter la longitude des maisons de notre dataset

mapcal = folium.Map(location=[latitude,longitude], zoom_start=5)
mapcal



for index, i  in house.iterrows():
    # representation des cercles bleu sur notre map 
    folium.Circle(
    radius=50,
    location=[i["lat"], i["long"]],
    
    color="blue",
    fill=False,
).add_to(mapcal)

mapcal



# In[4]:


house.head()


# Ce projet aura pour objectif de répondre à la problèmatique suivante:
# 
# **Comment évolue le prix de vente des maisons du comté de King à Seatle et quel modèle de prédiction peut-on mettre en place ?**
# 
# Nous réaliserons d'abord une **analyse générale** du prix de vente des maisons. Nous mettrons ensuite en place des **modèles de prédiction** en testant la précision de ceux-ci.

# # Analyse général du prix de vente des maisons

# ## Traitement des données

# In[5]:


# Renommage de la colonne Price par "PrixVente"

house.rename(columns ={'price': 'PrixVente'}, inplace =True)


# In[6]:


# Supression de la colonne ID et Date

house.drop(['id','date'],axis = 1, inplace = True)


# In[7]:


house.isna().sum()  # la fonction isna() permet de voir s'il y a des valeurs manquantes


# Notre data frame ne contient aucune donnée manquante.

# In[8]:


house.isnull().sum()  # la fonction isna() permet de voir s'il y a des valeurs nulles


# Notre data frame ne contient aucune donnée nulle.

# ## Répartition du prix de vente

# **Nous allons tout d'abord voir où sont situées les maisons dont le prix de vente est supérieur à la moyenne des prix.**

# In[9]:


m = house['PrixVente'].mean()


# In[10]:


# critere de selection : maisons dont le prix est superieur au prix minimum.
house_price = house[house["PrixVente"] > m]


for index, i  in house_price.iterrows():
   
    folium.Circle(
    radius=20,
    location=[i["lat"], i["long"]],
    
    color="red",
    fill=False,
).add_to(mapcal)

mapcal


# On observe sur la carte que les maisons dont le prix est supérieur à la moyenne sont concentrées principalement sur les bords de mer. 
# **On peut donc emettre l'hypothèse que le prix de vente augmente en fonction de la localisation,vue sur la mer (waterfront), des maisons.**

# In[11]:


# Histogramme des prix de ventes des maisons : 
plt.figure(figsize=(7, 5))

plt.hist(house['PrixVente'],
         bins=1000,
         color = 'red');

plt.xlabel('Prix de Vente');


# Le prix de vente s'écarte d'une distribution normale. 

# ## Quelles corrélations entre le prix de vente et les autres variables ?  

# In[12]:


corrmat = house.corr(method='pearson')

f, ax = plt.subplots(figsize=(8, 6))

sns.heatmap(corrmat, vmax=1, square=True);


# In[13]:


corrmat.style.background_gradient(cmap='coolwarm')


# ## Prix de vente en fonction d'autres variables

# Nous nous intéresserons aux variables qui semblent être le plus corrélés avec notre variable cible qui est "PrixVente".
# Grace à la matrice de corrélation ci dessus : nous prendrons les variables dont le coefficient est supérieur à 0.5, à savoir: 
#  
#  - grade (0.67)
#  - Bathrooms (0.53)
#  - sqft_living (0.70)
#  - sqft_living15 (0.59)
#  - sqft_above (0.61)
# 
# En revanche, certaines variables sont très peu corrélées :
# 
#  - Yr_built (0.054)
#  - sqft_lot15 (0.082)

# Définissons une fonction affichant la variable **Prix de Vente** en fonction d'une autre.

# In[14]:


def relation_pdv(variable):
    
    data = pd.concat([house['PrixVente'],
                      house[variable]],
                     axis=1)
    
    data.plot.scatter(x=variable,
                      y='PrixVente',
                      c='red', );


# ### Les variables fortement corrélées

# In[15]:


relation_pdv('grade')

relation_pdv('bathrooms')

relation_pdv('sqft_living')

relation_pdv('sqft_above')

relation_pdv('sqft_living15')


# Ces graphes affichent  le prix de vente en fonction des variables les plus corrélées.
# Il est clair que le prix de vente évolue **linéairement** en à mesure que ces variables augmentent.

# ### Les variables peu corrélées

# In[16]:


relation_pdv('yr_built')

relation_pdv('sqft_lot15')


# Ces graphe affichant le prix de vente en fonctions des variables les moins corrélées. En particulier, les maisons les plus récentes ne sont pas nécessairement les plus coûteuses comme indique **l'abscence de relation linéaire** sur le graphe ci-dessus.
# 
# De plus notre hypothèse de depart semble également fausse,**il n'y a pas une forte corrélation entre le prix de vente et waterfront ( maison avec vue sur mer) ou latitude et longitude.**

# # Machine Learning 

# ### Importation

# In[17]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import mean_squared_error, make_scorer
from scipy.stats import skew
from IPython.display import display
import matplotlib.pyplot as plt


# In[18]:


# Preparation de la base de données

variables = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors','view', 'condition', 'grade', 'sqft_above',
             'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15']

X = house[variables]

# Variable cible

Y = pd.DataFrame(house.PrixVente)

print(X.head())
print(Y.head())


# In[19]:


# On choisit de prendre une taille de 20% pour l'échantillion test et 80% pour l'échantillon d'entrainement

X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size= 0.2, random_state=42)


# In[20]:


print(X_train.shape)


# In[21]:


X_test.shape # 20%


# In[22]:


y_train.shape


# In[23]:


y_test.shape


# ## Normalisation centrée réduite de nos données 

# Avant d'appliquer nos algorithmes, nous allons ramener nos données a la meme echelle, nous allons calculer la moyenne et l'ecart type uniquement sur le dataset d'entrainement.
# Le scaler **StandardScaler** dans la librairie scikit-learn permet de realisation facilement la normalisation.

# In[24]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler() #creation du scaler 
scaler.fit(X_train) # calcul mu et sigma sur X_train


# In[25]:


print('Moyenne mu',scaler.mean_)
print('Ecart-type sigma',scaler.scale_)


#  Appliquons les valeurs $mu$ et $sigma$ calculées pour X_train aux deux datasets X_train et X_test 

# In[26]:


X_train_scale = scaler.transform(X_train) # numpy 
X_train_scale = pd.DataFrame(X_train_scale,index = X_train.index) # dataframe


# In[27]:


X_test_scale = scaler.transform(X_test) # numpy 
X_test_scale = pd.DataFrame(X_test_scale,index = X_test.index)


# In[28]:


X_train_scale.mean().head()


# Les moyennes sont des valeurs tres petites proche de 0.

# In[29]:


X_train_scale.std().head()


# Nous avons bien 1 en ecart type. nos données sont désormais normalisées.

# ## Classifications: decision tree, randomforest, SVM*

# In[30]:


X_test.shape


# In[31]:


X_train.shape


#  Nous allons souvent réaliser les étapes d'entrainement, de prédiction et de calcul de la métrique pour différents modèles de machine learning. Il serait plus pratique de créer une fonction: calcul_accuracy

# In[32]:


from sklearn import metrics
def calcul_accuracy(classifier, X_train, X_test, y_train, y_test):
    classifier.fit(X_train, y_train)
    y_pred_train = classifier.predict(X_train)
    y_pred_test = classifier.predict(X_test)
    accuracy_train = metrics.accuracy_score(y_train, y_pred_train)
    accuracy_test = metrics.accuracy_score(y_test, y_pred_test)
    print('Train accuracy:', '{:.2f}'.format(accuracy_train), 'Test accuracy:', '{:.2f}'.format(accuracy_test))
    return accuracy_train, accuracy_test, classifier


# In[33]:


# Entrainement du Decision tree

from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(max_depth=1, random_state=42, criterion='entropy')
classifier.fit(X_train, y_train)




# In[34]:


from sklearn.tree import export_text, plot_tree

decision_tree_text = export_text(classifier,feature_names=list(X_train.columns))
print(decision_tree_text)


# In[35]:


plot_tree(classifier,feature_names=list(X_train.columns), precision=2, filled=True)


# On a un seuil de decision au niveau de la variable **lat** qui est de 47.53. Si la latitude est inférieur ou égale à 47.53  nous avons 6621 échantillions. Si c'est superieur à 47.53, on a un échantillion de 10669.

# In[36]:


calcul_accuracy(classifier, X_train, X_test, y_train, y_test)


# Cette classification n'est pas fiable du tout ( seulement 1% d'accuracy).
# Nous avons utilisé les données normalisées et nous obtenons également 1% d'accuracy 
# 

# In[37]:


# Entrainement du  RandomForest avec 20 arbres
classifier1 = RandomForestClassifier(max_depth=2, n_estimators=20, random_state=42, criterion='entropy')
accuracy_train, accuracy_test, trained_classifier1 = calcul_accuracy(classifier1, X_train, X_test, y_train, y_test)


# In[38]:


# Matrice de confusion 
#metrics.plot_confusion_matrix(trained_classifier1, X_test, y_test)
# ne fonctionne pas 


# Meme constat pour le modele de classification Randomforest, on avons 1% d'accuracy, ce qui signifie que notre modele n'est pas fiable.

# ## Un modèle linéaire de Support Vector Machine (SVM)

# Le modèle SVM cherche à définir une frontière entre deux ou plusieurs classes d'échantillions, en maximisant la marge entre cette frontiere et lees echantillions les plus proches qu'on appelle les vecteurs de support.

# In[39]:


from sklearn.svm import LinearSVC
classifier = LinearSVC(random_state=42)


# In[ ]:


# Avec les données normalisées 
calcul_accuracy(classifier, X_train_scale, X_test_scale, y_train, y_test)


# Le modele SVM est lui aussi mauvais.

# Nous avons obtenu 3 modeles de classifications avec 1% d'accuracy, 
# nos données ne peuvent pas être entrainer  dans les modeles multi-classes de classification.

# ## Régression linéaire

# Définissons deux fonctions donnant les valeurs des RMSE pour chacun des échantillons d'entraînement et de test.

# In[48]:


def rmse_train(pred):
    return np.sqrt(mean_squared_error(y_train, pred))

def rmse_test(pred):
    return np.sqrt(mean_squared_error(y_test, pred))


# In[49]:


# Création du modèle 
reglin = LinearRegression()

# entrainement du modèle 
reglin.fit(X_train_scale, y_train)


# In[50]:


# Prédiction à l'aide du modèle linéaire
y_pred = reglin.predict(X_test_scale)


# In[51]:


rmse_test(y_pred)


# In[52]:


print("Accuracy: " + str(reglin.score(X_test_scale, y_test)))


# Le modèle linéaire obtient un score de **68%** en terme de précision de prédiction. Nous comparerons la RMSE avec celles des prochains modèles.
# 
# Nous avons en parallèle effectué la régression linéaire en ne gardant que les variables les plus corrélées. Nous obtenons une précision de prédiction de **55%**. C'est pourquoi nous garderons dans la suite l'ensemble des variables.

# ## Régression Ridge

# Dans ce modèle de prédiction, nous réalisons une régression linéaire Ridge du prix de vente, c'est à dire avec une régularisation L2.

# Réalisons la régression Ridge et affichons les valeurs de l'accuracy et de la RMSE.

# In[53]:


ridge = RidgeCV()
ridge.fit(X_train_scale, y_train)

y_train_pred = ridge.predict(X_train_scale)
y_test_pred = ridge.predict(X_test_scale)

print("Valeur de la RMSE sur les données d'entraînement :", rmse_train(y_train_pred))
print("Valeur de la RMSE sur les données test :", rmse_test(y_test_pred))


# In[54]:


print("Accuracy: " + str(ridge.score(X_test_scale, y_test)))


# Le modèle Ridge obtient un score de **68%** en terme de précision de prédiction comme c'est le cas avec le modèle linéaire. En revanche, la RMSE est légèrement plus élevée.

# Affichons en complément les graphes des valeurs prédites ainsi que des résidus.

# In[55]:


# Graphe des prédicitions

plt.scatter(y_train_pred,
            y_train,
            c = "blue",
            marker = ".",
            label = "Données d'entraînement")

plt.scatter(y_test_pred,
            y_test,
            c = "red",
            marker = ".",
            label = "Données test")

plt.title("Régression linéaire Ridge")
plt.xlabel("Valeurs prédites")
plt.ylabel("Valeurs réelles")
plt.legend(loc = "upper left")
plt.show()

# Graphe des résidus

plt.scatter(y_train_pred,
            y_train_pred - y_train,
            c = "blue",
            marker = ".",
            label = "Données d'entraînement")

plt.scatter(y_test_pred,
            y_test_pred - y_test,
            c = "red",
            marker = ".",
            label = "Données test")

plt.title("Régression linéaire Ridge")
plt.xlabel("Valeurs prédites")
plt.ylabel("Résidus")
plt.legend(loc = "upper left")
plt.show()


# La variable prenant le plus d'importance dans la régression Ridge effectué est la variables **lat**. C'est cette variable qui permet la meilleur prédiction du prix de vente.

# ## Conclusion 

# Nous avons mis en place des modèles de classification qui ont été mediocre dans le cas de nos données puisque nous avons obtenu 1% en terme de précision de prédiction.
# 
# En revanche pour les modeles de Regression linéaire et Ridge, nous avons une precision de prédiction à 68 % pour les 2 modeles, ce qui est bien.
