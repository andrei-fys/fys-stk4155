from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from sklearn.linear_model import LinearRegression,RidgeCV,Lasso
from sklearn.metrics import mean_squared_error, r2_score
from Franke import FrankeFunction, SetUpDesignMat, SetUpGrid

def ScikitSolverOLS(data, z):
    print(' ============== Scikit OLS ============')
    clf5 = LinearRegression(fit_intercept=False)
    clf5.fit(data,z)
    x_n, y_n = SetUpGrid(N)
    x_n = x_n.reshape(-1, 1)
    y_n = y_n.reshape(-1, 1)
    data_n = SetUpDesignMat(x_n,y_n,N)
    z_n = clf5.predict(data_n)
    R2 = clf5.score(data_n, z.reshape(-1, 1))
    MSE = mean_squared_error(z.reshape(-1, 1), z_n)
    print('R2 SciKit', R2)
    print('MSE SciKit  ',MSE)
    print('Coefficient beta : \n', clf5.coef_)

def ScikitSolverRidge(data, z):
    print(' ============== Scikit Ridge ============')
    ridge=RidgeCV(alphas=[0.1,1.0,10.0])
    ridge.fit(data,z)
    print("Ridge Coefficient: ",ridge.coef_)
    print("Ridge Intercept: ", ridge.intercept_)
    
def ScikitSolverLasso(data, z):
    print(' ============== Scikit Lasso ============')
    lasso=Lasso(alpha=0.1)
    lasso.fit(data,z)
    predl=lasso.predict(data)
    print("Lasso Coefficient: ", lasso.coef_)
    print("Lasso Intercept: ", lasso.intercept_)

def NaiveSolverOLS(data, z):
    print(' ============== Naive OLS ============')	
    xb = data
    y = z
    beta = np.linalg.inv(xb.T.dot(xb)).dot(xb.T).dot(y)
    x_n, y_n = SetUpGrid(N)
    x_n = x_n.reshape(-1, 1)
    y_n = y_n.reshape(-1, 1)
    data_n = SetUpDesignMat(x_n,y_n,N)
    ypredict = data_n.dot(beta)
    print('Coefficient beta naive: \n', beta.reshape(1,-1))
    print('Mean naive ', Mean(y))
    print('R2 naive ', R2(y, ypredict))
    print('MSE naive  ', MSE(y, ypredict))

def NaiveSolverRidge(data, z):	
    xb = data
    y = z
    beta = np.linalg.inv(xb.T.dot(xb) + lambdaI).dot(xb.T).dot(y)
    x_n, y_n = SetUpGrid(N)
    x_n = x_n.reshape(-1, 1)
    y_n = y_n.reshape(-1, 1)
    data_n = SetUpDesignMat(x_n,y_n,N)
    ypredict = data_n.dot(beta)
    print('Coefficient beta naive Ridge: \n', beta.reshape(1,-1))
    print('Mean naive ', Mean(y))
    print('R2 naive ', R2(y, ypredict))
    print('MSE naive  ', MSE(y, ypredict))


def Mean(y):
    return y.mean(axis=0)

def R2(y, y_predict):
    assert len(y)==len(y_predict)
    n = len(y)
    y_mean = Mean(y)
    s1, s2 = 0, 0
    for i in range(0, n):
        s1 += (y[i] - y_predict[i])**2
        s2 += (y[i] - y_mean)**2
    s = s1/s2    
    return 1 - s

def MSE(y, y_predict):
    assert len(y)==len(y_predict)
    n = len(y)
    s = 0
    for i in range(0, n):
        s += (y[i] - y_predict[i])**2
    s /=float(n)
    return s




#Number of grid points in one dim
N = 20
#Define interval for the arguments of the Franke function
x_exact = np.arange(0, 1, 0.05)
y_exact = np.arange(0, 1, 0.05)
#Round round baby round round 
x, y = SetUpGrid(N)
z = FrankeFunction(x,y)
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)
z = z.reshape(N*N,1)
data = SetUpDesignMat(x,y,N)

ScikitSolverRidge(data, z)
ScikitSolverLasso(data, z)
#print("########################################")
#print("Naive: ")
NaiveSolverOLS(data, z)
#print("Scikit: ")
ScikitSolverOLS(data, z)
