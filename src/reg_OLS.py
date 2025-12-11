import numpy as np
import matplotlib.pyplot as plt
#from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

# Fermeture des plots
plt.close('all')

########################################
# Regression OLS

# 1. Avec data1.npy
data1 = np.load("../data/data1.npy")

X = data1[0]
Y = data1[1]

vect_one = np.ones(len(X))

X_ = np.c_[vect_one, X] 

beta = np.linalg.inv(X_.T.dot(X_)).dot(X_.T).dot(Y)

f = beta[0] + beta[1]*X

erreur = np.mean((f - Y)**2)
print("Erreur d'apprentissage: ", erreur)

# Affichage
#plt.scatter(X, Y)
#plt.plot(X, f)

# 2. Avec data2.npy
# .......

############################################
# Regression polynomiale: espace de redescription

# Fonction pour factoriser la régression OLS
def OLS(X,Y):
    return np.linalg.inv(X.T @ X) @ X.T @ Y

# Fonction de regression polynomiale
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

# 4. OLS avec data2.npy et modèle polynomial
# Chargement des données
data2 = np.load("../data/data2.npy")
X2 = data2[0]
Y2 = data2[1]

# Test fonction reg_poly
for q in range(1,11):       # q est l'ordre de la régression polynomiale
    X_reg, Y_reg, e = reg_poly(data2, q)
    #e_ = round(e, 4)
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
# Approche Ridge
# Fonction

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

# 7. Extraction des données
# On reprend data3
q = 10
lambdas = [0.01, 0.1, 1., 10., 20., 100.]

plt.figure()
for lambd in range(0, len(lambdas)):
    X3_reg_ridge, Y3_reg_ridge, e3 = reg_Ridge(data3, q, lambdas[lambd])
    plt.plot(X3_reg_ridge, Y3_reg_ridge, label=f"Ordre: {q}, lambda: {lambdas[lambd]}, erreur: {e3:.4f}")

# Affichage des données
plt.scatter(X3, Y3, color='blue', label='Data3')
plt.title("Regression Ridge sur data3, ordre 10")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()





