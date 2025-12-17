###############################################################################
# MODULES
###############################################################################
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import scipy.linalg as la
import numpy.random as rnd
###############################################################################

################################################################################
# PARAMETERS
################################################################################
# Dimension and sample size
p=2
n=600
# Proportion of sample from classes 0, 1, and outliers
p0 = 3/6
p1 = 2/6
pout = 1/6
# Examples of means/covariances of classes 0, 1 and outliers
mu0 = np.array([-2,-2])
mu1 = np.array([2,2])
muout = np.array([-8,-8])
Sigma_ex1 = np.eye(p)
Sigma_ex2 = np.array([[5, 0.1],
                      [1, 0.5]])
Sigma_ex3 = np.array([[0.5, 1],
                      [1, 5]])
Sigma0 = Sigma_ex1
Sigma1 = Sigma_ex1
Sigmaout = Sigma_ex1
# Regularization coefficient
lamb = 0
################################################################################

################################################################################
# DATA/LABELS GENERATION
################################################################################
# Sample sizes
n0 = int(np.floor(n*p0))
n1 = int(np.floor(n*p1))
nout = int(np.floor(n*pout))
if n0+n1+nout < n:
   n0 += int(n - (n0+n1+nout))
# Data and labels
mu0_mat = mu0.reshape((p,1))@np.ones((1,n0))
mu1_mat = mu1.reshape((p,1))@np.ones((1,n1))
x0 = np.zeros((p,n0+nout))
x0[:,0:n0] = mu0_mat + la.sqrtm(Sigma0)@rnd.randn(p,n0)
x1 = mu1_mat + la.sqrtm(Sigma1)@rnd.randn(p,n1)
if nout > 0:
  muout_mat = muout.reshape((p,1))@np.ones((1,nout))
  x0[:,n0:n0+nout] = muout_mat + la.sqrtm(Sigmaout)@rnd.randn(p,nout)
y = np.concatenate((-np.ones(n0+nout),np.ones(n1)))
X = np.ones((n,p+1))
for i in np.arange(n):
     X[0:n0+nout,1:p+1] = x0.T
     X[n0+nout:n,1:p+1] = x1.T
################################################################################
# Regression Logistique
regression_logistique = LogisticRegression(penalty='l2')

# Entraînement du modèle
#on enlève le biais
regression_logistique.fit(X[:,1:], y)

################################################################################
# PLOTS
################################################################################
#fig,ax = plt.subplots()
#ax.plot(x0[0,:],x0[1,:],'xb',label='Class 0')
#ax.plot(x1[0,:],x1[1,:],'xr',label="Class 1")
#ax.legend(loc = "upper left")

# Frontière de décision
x_min, x_max = X[:,1].min()-1, X[:,1].max()+1
y_min, y_max = X[:,2].min()-1, X[:,2].max()+1
zone_x, zone_y = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

#Prédictions
Z = regression_logistique.predict(np.c_[zone_x.ravel(), zone_y.ravel()])
Z = Z.reshape(zone_x.shape)

fig, ax = plt.subplots()
ax.contourf(zone_x, zone_y, Z, alpha=0.8, cmap=plt.cm.coolwarm)
#Tracé des points
ax.plot(x0[0, :], x0[1, :], 'xb', label='Class 0')
ax.plot(x1[0, :], x1[1, :], 'xr', label="Class 1")


#hyperplan de décision
coef = regression_logistique.coef_[0]
intercept = regression_logistique.intercept_[0]

#Tracer de la frontière de décision
x_min_plot = min(x0[0,0:n0].min(), x1[0,:].min())
x_max_plot = max(x0[0,0:n0].max(), x1[0,:].max())

x_axis = np.linspace(x_min_plot, x_max_plot, 50)
y_logreg = -(coef[0]*x_axis + intercept)/coef[1]

ax.plot(x_axis, y_logreg, '-k', label='Frontière de décision')
ax.legend(loc="upper left")
ax.set_title("Régression Logistique")
plt.grid()
plt.show()