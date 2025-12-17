###############################################################################
# MODULES
###############################################################################
import numpy as np
import matplotlib.pyplot as plt
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
lamb = 10000
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
# Classification: OLS AND RIDGE

# OLS 
XtX = X.T @ X # X transposé fois X
Xty = X.T @ y # X transposé fois y
beta_ols = np.linalg.inv(XtX) @ Xty # Coefficients de la régression OLS


# Ridge 
I = np.eye(X.shape[1]) #Matrice identité
I[0,0] = 0 
beta_ridge = np.linalg.inv(XtX + lamb*I) @ Xty # Coefficients de la régression Ridge



#Pour verifier que les predictions fonctionnent
def predict_ols(X, beta):
   return np.sign(X @ beta)
#y_pred = predict_ols(X, beta_ols)
#print(y_pred[:550])


################################################################################
# PLOTS
################################################################################
fig,ax = plt.subplots()
ax.plot(x0[0,:],x0[1,:],'xb',label='Class 0')
ax.plot(x1[0,:],x1[1,:],'xr',label="Class 1")
ax.legend(loc = "upper left")
#plt.show()


# Hyperplan OLS
x_axis = np.linspace(-10,10,100) #géneration de points
y_ols = -(beta_ols[1]*x_axis + beta_ols[0])/beta_ols[2]
ax.plot(x_axis,y_ols,'--y',label='Hyperplan OLS') 
ax.legend(loc = "upper left")
#plt.show()


# Hyperplan Ridge 
y_ridge = -(beta_ridge[1]*x_axis + beta_ridge[0])/beta_ridge[2]
ax.plot(x_axis,y_ridge,'--k',label='Hyperplan Ridge') 
ax.legend(loc = "upper left")
ax.set_xlim([-10,10])
ax.set_ylim([-10,10])
ax.set_title("Calassification OLS et Ridge")
plt.grid()
plt.show()



