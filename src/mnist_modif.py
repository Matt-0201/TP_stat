###############################################################################
# MODULES
###############################################################################
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
###############################################################################
# LOAD MNIST
###############################################################################
# Download MNIST
mnist = fetch_openml(data_id=554, parser='auto')
# copy mnist.data (type is pandas DataFrame)
data = mnist.data
# array (70000,784) collecting all the 28x28 vectorized images
img = data.to_numpy()
# array (70000,) containing the label of each image
lb = np.array(mnist.target,dtype=int)
# Splitting the dataset into training and test subsets
X_train, X_test, y_train, y_test = train_test_split(
    img, lb, 
    test_size=0.25, 
    random_state=0)
# Number of classes
k = len(np.unique(lb))
# Sample sizes and dimension
(n,p) = img.shape
n_train = y_train.size
n_test = y_test.size 
###############################################################################
# DISPLAY A SAMPLE
###############################################################################
m=16
plt.figure(figsize=(10,10))
for i in np.arange(m):
  ex_plot = plt.subplot(int(np.sqrt(m)),int(np.sqrt(m)),i+1)
  plt.imshow(img[i,:].reshape((28,28)), cmap='gray')
  ex_plot.set_xticks(()); ex_plot.set_yticks(())
  #lt.title("Label = %i" % lb[i])
  
###############################################################################
# CLASSIFICATION MULTI-CLASSES SUR MNIST
###############################################################################

# Normalisation des données
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train.astype(float))
X_test_sc = sc.transform(X_test.astype(float))

# On lance la logistique en L2
lr_l2 = LogisticRegression(penalty="l2", solver="saga")
lr_l2.fit(X_train_sc, y_train)

# Calcul des prédictions et affichage de la matrice
y_pred = lr_l2.predict(X_test_sc)
conf_mat = confusion_matrix(y_test, y_pred)

plt.figure()
ConfusionMatrixDisplay(conf_mat, display_labels=np.unique(lb)).plot(cmap='Blues', ax=plt.gca())
plt.title("Matrice de confusion (L2)")
plt.show()

# 2. Visiualisation des betas (carte de couleur RdBu)
betas = lr_l2.coef_

fig, axes = plt.subplots(2, 5, figsize=(12, 6))
axes = axes.flatten()

for i in range(10):
    # On reshape les coefficients en 28x28 pour retrouver l'image
    b_img = betas[i].reshape(28, 28)
    # On centre l'échelle de couleur pour que le blanc soit à 0
    vmax = np.max(np.abs(b_img))
    axes[i].imshow(b_img, cmap='RdBu', vmin=-vmax, vmax=vmax)
    axes[i].set_title(f"Chiffre {i}")
    axes[i].axis('off')

plt.tight_layout()
plt.show()

# 3. Régression logistique en l1 (Lasso)
lr_l1 = LogisticRegression(penalty="l1", solver="saga", tol=0.1)
lr_l1.fit(X_train_sc, y_train)

# Matrice de confusion pour le Lasso
y_pred_l1 = lr_l1.predict(X_test_sc)
conf_mat_l1 = confusion_matrix(y_test, y_pred_l1)

plt.figure()
ConfusionMatrixDisplay(conf_mat_l1, display_labels=np.unique(lb)).plot(cmap='Greens', ax=plt.gca())
plt.title("Matrice de confusion (L1)")
plt.show()

# --- Visualisation des Beta L1 (Comparaison avec L2) ---

betas_l1 = lr_l1.coef_

fig, axes = plt.subplots(2, 5, figsize=(12, 6))
axes = axes.flatten()

for i in range(10):
    b_img = betas_l1[i].reshape(28, 28)
    
    # Même principe : on centre l'échelle sur 0
    vmax = np.max(np.abs(b_img))
    
    axes[i].imshow(b_img, cmap='RdBu', vmin=-vmax, vmax=vmax)
    axes[i].set_title(f"Lasso - Chiffre {i}")
    axes[i].axis('off')

plt.tight_layout()
plt.show()
