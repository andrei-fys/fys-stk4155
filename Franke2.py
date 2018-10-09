from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from sklearn.linear_model import LinearRegression, RidgeCV, Lasso, Ridge 
from sklearn.metrics import mean_squared_error, r2_score 
import sklearn.preprocessing
import copy as cp
from functions import bias2, mse, ridge_regression_variance
import sklearn.model_selection 
import sys
#np.set_printoptions(threshold=np.nan)


def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4


def SetUpGrid(N):    
    x = np.sort(np.random.uniform(0,1,N))
    y = np.sort(np.random.uniform(0,1,N))
    x = np.sort(x)
    y = np.sort(y)
    x, y = np.meshgrid(x,y)
    return x, y

def SetUpDesignMatrix(degree, x,y):
    polynom = sklearn.preprocessing.PolynomialFeatures(degree=degree, include_bias=True)
    X = polynom.fit_transform(cp.deepcopy(np.c_[x.reshape(-1, 1), y.reshape(-1, 1)]))
    return X

def OLS_SK(degree, X):
    olsreg = LinearRegression(fit_intercept=False)
    z = FrankeFunction(x,y)
    olsreg.fit(X, z.ravel())
    zpredict = olsreg.predict(X)
    r2 = r2_score(z.ravel(), zpredict)
    bias = bias2(z.ravel(), zpredict)
    mse_error = mse(z.ravel(), zpredict)
    N, P = X.shape
    z_variance = np.sum((z.ravel() - zpredict)**2) / (N - P - 1)
    linreg_coef_var = np.diag(np.linalg.inv(X.T @ X))*z_variance
    print ('Variance in beta for OSL ',linreg_coef_var)
    print ('R2, Bias, MSE error for OSL ',r2,bias,mse_error)


def Rigde_SK(degree, X, alpha):
    ridge = Ridge(alpha=alpha, solver="lsqr", fit_intercept=False)
    z = FrankeFunction(x,y)
    noise_val = np.random.normal(0,1,np.size(z))
    ridge.fit(X, z.ravel())
    zpredict = ridge.predict(X)
    r2 = r2_score(z.ravel(), zpredict)
    R2 = ridge.score(X, z.ravel())
    bias = bias2(z.ravel(), zpredict)
    mse_error = mse(z.ravel(), zpredict)
    print (r2,bias,mse_error)
    N, P = X.shape
    z_variance = np.sum((z.ravel() - zpredict)**2) / (N - P - 1)
    beta_variance = ridge_regression_variance(
            X, z_variance, alpha)
    print ('Variance in beta for Ridge ',beta_variance)
    print ('R2, Bias, MSE error for Ridge ',r2, bias,mse_error)
    print ('Alternative r2 ',R2)


    
def Lasso_SK(degree, X, alpha):
    lasso = Lasso(alpha=alpha, fit_intercept=False)
    z = FrankeFunction(x,y)
    lasso.fit(X, z.ravel())
    zpredict = lasso.predict(X)
    r2 = - r2_score(z.ravel(), zpredict)
    R2 = - lasso.score(X, z.ravel())
    bias = bias2(z.ravel(), zpredict)
    mse_error = mse(z.ravel(), zpredict)
    print (r2,bias,mse_error)
    N, P = X.shape
    z_variance = np.sum((z.ravel() - zpredict)**2) / (N - P - 1)
    beta_variance = ridge_regression_variance(
            X, z_variance, alpha)
    print ('Variance in beta for Lasso ',beta_variance)
    print ('R2, Bias, MSE error for Lasso ',r2, bias,mse_error)
    print ('Alternative r2 ',R2)






if __name__ == '__main__':

    N=1000
    degree = int(sys.argv[1])
    #noise = sys.argv[2]
    #degree = 5
    x,y = SetUpGrid(N);
    X = SetUpDesignMatrix(degree, x,y)
    #print (np.size(X))
    OLS_SK(degree, X)
    alpha = float(sys.argv[2])
    #alpha = 0.2
    Rigde_SK(degree, X, alpha)
    Lasso_SK(degree, X, alpha)
    
