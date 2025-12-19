import numpy as np
import matplotlib.pyplot as plt
#from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso

# Fermeture des plots
plt.close('all')

############################################################################### 

# Fonctions principales

###############################################################################

## Foncion pour les moindres carrés
def OLS(X,Y):
    return np.linalg.inv(X.T @ X) @ X.T @ Y

## Fonction de regression polynomiale
def reg_poly(data, degree):
    X = data[0]
    Y = data[1]
    
    # Espace polynomial
    poly = PolynomialFeatures(degree=degree)     # Pas besoin d'ajouter le biais manuellement, PolynomialFeatures le fait par défaut
    X_poly = poly.fit_transform(X.reshape(-1, 1))
    
    # Regression OLS
    beta = OLS(X_poly, Y)
    
    # Création d'un espace 
    X_reg = np.linspace(np.min(X), np.max(X), 100).reshape(-1, 1)
    X_reg_poly = poly.transform(X_reg)
    
    # Prédiction
    Y_reg_poly = X_reg_poly @ beta
    
    # Erreur
    f = X_poly @ beta
    e = np.mean((f - Y)**2)
    
    return X_reg, Y_reg_poly, e

## Fonction pour l'approche Ridge
def reg_Ridge(data, degree, lambd):
    X = data[0]
    Y = data[1]
    
    # Espace polynomial
    poly = PolynomialFeatures(degree=degree)     # Pas besoin d'ajouter le biais manuellement, PolynomialFeatures le fait par défaut
    X_poly = poly.fit_transform(X.reshape(-1, 1))
    
    # Regression OLS
    modele_ridge = Ridge(alpha=lambd)
    modele_ridge.fit(X_poly, Y)
    
    # Création d'un espace 
    X_reg = np.linspace(np.min(X), np.max(X), 100).reshape(-1, 1)
    X_reg_poly = poly.transform(X_reg)
    
    # Prédiction
    Y_reg_poly = modele_ridge.predict(X_reg_poly)
    
    # Erreur
    f = modele_ridge.predict(X_poly)
    e = np.mean((f - Y)**2)
    
    return X_reg, Y_reg_poly, e

## Fonction pour l'approche Lasso
def reg_Lasso(data, degree, lambd):
    X = data[0]
    Y = data[1]
    
    # Espace polynomial
    poly = PolynomialFeatures(degree=degree)     # Pas besoin d'ajouter le biais manuellement, PolynomialFeatures le fait par défaut
    X_poly = poly.fit_transform(X.reshape(-1, 1))
    
    # Regression OLS
    modele_lasso = Lasso(alpha=lambd)
    modele_lasso.fit(X_poly, Y)
    
    # Création d'un espace 
    X_reg = np.linspace(np.min(X), np.max(X), 100).reshape(-1, 1)
    X_reg_poly = poly.transform(X_reg)
    
    # Prédiction
    Y_reg_poly = modele_lasso.predict(X_reg_poly)
    
    # Erreur
    f = modele_lasso.predict(X_poly)
    e = np.mean((f - Y)**2)
    
    return X_reg, Y_reg_poly, e

###############################################################################

# Code de tests

###############################################################################
# Regression OLS

# 1. Avec data1.npy
data1 = np.load("../data/data1.npy")

X1 = data1[0]
Y1 = data1[1]

# Vecteur de 1 pour le biais
vect_one = np.ones(len(X1))

# Ajout du biais sur la matrice
X1_ = np.c_[vect_one, X1] 

beta = np.linalg.inv(X1_.T.dot(X1_)).dot(X1_.T).dot(Y1)

f = beta[0] + beta[1]*X1

erreur = np.mean((f - Y1)**2)
print("Erreur d'apprentissage: ", erreur)

# Affichage 
plt.figure()
plt.plot(X1, f, label=f"Erreur: {erreur:.4f}");
plt.scatter(X1, Y1, color='blue', label='Data1')
plt.title("Regressions OLS sur data1")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()

# 2. Avec data2.npy
data2 = np.load("../data/data2.npy")

X2 = data2[0]
Y2 = data2[1]

# Vecteur de 1 pour le biais
vect_one = np.ones(len(X2))

# Ajout du biais sur la matrice
X2_ = np.c_[vect_one, X2] 

beta = np.linalg.inv(X2_.T.dot(X2_)).dot(X2_.T).dot(Y2)

f = beta[0] + beta[1]*X2

erreur = np.mean((f - Y2)**2)
print("Erreur d'apprentissage: ", erreur)

# Affichage 
plt.figure()
plt.scatter(X2, Y2, color='blue', label='Data2')
plt.plot(X2, f, label=f"Erreur: {erreur:.4f}")
plt.title("Regressions OLS sur data2")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()

############################################
# Regression polynomiale: espace de redescription
############################################

# 4. OLS avec data2.npy et modèle polynomial
# Chargement des données
data2 = np.load("../data/data2.npy")
X2 = data2[0]
Y2 = data2[1]

# Test fonction reg_poly
plt.figure()
for q in range(1,11):       # q est l'ordre de la régression polynomiale
    X_reg, Y_reg, e = reg_poly(data2, q)
    plt.plot(X_reg, Y_reg, label=f"Ordre: {q}, erreur: {e:.4f}");

# Affichage 
plt.scatter(X2, Y2, color='blue', label='Data2')
plt.title("Regressions polynomiales sur data2, ordre 1-10")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()

# 5. Avec data3.npy
# Extraction des données
data3 = np.load("../data/data3.npy")
X3 = data3[0]
Y3 = data3[1]
q = 10

# Regression polynomiale
X3_reg_lin, Y3_reg_lin, e3 = reg_poly(data3, q)

# Affichage des résultats
plt.figure()
plt.plot(X3_reg_lin, Y3_reg_lin, color="red", label=f"Ordre: {q}, erreur: {e3:.4f}")
plt.scatter(X3, Y3, color='blue', label='Data3')
plt.title("Regression polynomiale sur data3, ordre 10")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()

#########################################

# 7. Régression avec Ridge()
q = 10
lambdas = [0.01, 0.1, 1., 10., 20., 100.]

plt.figure(figsize=[12, 8])
for lambd in range(0, len(lambdas)):
    X3_reg_ridge, Y3_reg_ridge, e3 = reg_Ridge(data3, q, lambdas[lambd])
    plt.plot(X3_reg_ridge, Y3_reg_ridge, label=f"Ordre: {q}, lambda: {lambdas[lambd]}, erreur: {e3:.4f}")

# Affichage des données
plt.scatter(X3, Y3, color='blue', label='Data3')
plt.title("Regression Ridge sur data3, ordre 10")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Régression avec Lasso
plt.figure()
for lambd in range(0, len(lambdas)):
    X3_reg_lasso, Y3_reg_lasso, e3 = reg_Lasso(data3, q, lambdas[lambd])
    plt.plot(X3_reg_lasso, Y3_reg_lasso, label=f"Ordre: {q}, lambda: {lambdas[lambd]}, erreur: {e3:.4f}")

# Affichage des données
plt.scatter(X3, Y3, color='blue', label='Data3')
plt.title("Regression Lasso sur data3, ordre 10")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()

# Comparaison des 3:
plt.figure(figsize=[10, 6])
l_cmp = 0.1
# 1. OLS
X_ols, Y_ols, e_ols = reg_poly(data3, q)
plt.plot(X_ols, Y_ols, '--', label=f"OLS, erreur: {e_ols:.4f}")

# 2. Ridge
X_ridge, Y_ridge, e_ridge = reg_Ridge(data3, q, l_cmp)
plt.plot(X_ridge, Y_ridge, label=f"Ridge (λ={l_cmp}), erreur: {e_ridge:.4f}", linewidth=2)

# 3. Lasso
X_lasso, Y_lasso, e_lasso = reg_Lasso(data3, q, l_cmp)
plt.plot(X_lasso, Y_lasso, label=f"Lasso (λ={l_cmp}), erreur: {e_lasso:.4f}", linewidth=2)

plt.scatter(X3, Y3, color='blue', label='Données réelles')
plt.title(f"Comparaison des méthodes OLS, Ridge et Lasso avec un ordre {q} et un lambda {l_cmp}")
plt.legend()
plt.show()




