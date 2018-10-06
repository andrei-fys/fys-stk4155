#!/usr/bin/env python
#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.pyplot as plt
#from matplotlib import cm
#from matplotlib.ticker import LinearLocator, FormatStrFormatter

import numpy as np
from random import random, seed
from sklearn.linear_model import LinearRegression,RidgeCV,Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from Franke1 import FrankeFunction, SetUpDesignMat, SetUpGrid

def ScikitSolverOLS(data, z, degree):
    clf5 = LinearRegression(fit_intercept=False)
    clf5.fit(data,z)
    x_n, y_n = SetUpGrid(N)
    x_n = x_n.reshape(-1, 1)
    y_n = y_n.reshape(-1, 1)
    data_n = SetUpDesignMat(x_n,y_n,N,degree)
    z_n = clf5.predict(data_n)
    R2 = clf5.score(data_n, z.reshape(-1, 1))
    MSE = mean_squared_error(z.reshape(-1, 1), z_n)
    print('R2 SciKit', R2)
    print('MSE SciKit  ',MSE)
    print('Coefficient beta : \n', clf5.coef_)

def ScikitSolverRidge(data, z):
    #ridge=RidgeCV(alphas=[0.1,1.0,10.0])
    ridge=Ridge(alpha=0.1, fit_intercept=False)
    ridge.fit(data,z)
    print("Ridge Coefficient: ",ridge.coef_)
    print("Ridge Intercept: ", ridge.intercept_)
    
def ScikitSolverLasso(data, z):
    lasso=Lasso(alpha=0.1)
    lasso.fit(data,z)
    predl=lasso.predict(data)
    print("Lasso Coefficient: ", lasso.coef_)
    print("Lasso Intercept: ", lasso.intercept_)

def NaiveSolverOLS(z, data, data_n, x_n, y_n):
    beta = np.linalg.inv(data.T.dot(data)).dot(data.T).dot(z)
    zpredict = data_n.dot(beta)
    #print('Coefficient beta naive: \n', beta.reshape(1,-1))
    beta = beta.reshape(-1,1)
    #print(' Beta0 ', beta[0])
    #print('Mean naive ', Mean(z))
    #print('R2 naive ', R2(z, zpredict))
    #print('MSE naive  ', MSE(z, zpredict))
    return beta

def NaiveSolverRidge(data, z, x_n, y_n, data_n, degree):
    I = np.identity(degree+1)	
    lambda_parameter = 0.1
    beta = np.linalg.inv(data.T.dot(data) + lambda_parameter*I).dot(data.T).dot(z)
    zpredict = data_n.dot(beta)
    print('Coefficient beta naive Ridge: \n', beta.reshape(1,-1))
    print('Mean naive ', Mean(z))
    print('R2 naive ', R2(z, zpredict))
    print('MSE naive  ', MSE(z, zpredict))
    return beta


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

def SetUpData(N, degree):
    x, y = SetUpGrid(N)
    z = FrankeFunction(x,y)
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    z = z.reshape(N*N,1)
    data = SetUpDesignMat(x,y,N,degree)
    x_n, y_n = SetUpGrid(N)
    x_n = x_n.reshape(-1, 1)
    y_n = y_n.reshape(-1, 1)
    data_n = SetUpDesignMat(x_n,y_n,N,degree)
    return z, data, data_n, x_n, y_n

def ConfidentIntervalBeta(Experiments, N, degree):
    betaBundle = np.zeros(shape=(degree+1,Experiments))
    for i in range (0, Experiments):
        z, data, data_n, x_n, y_n = SetUpData(N,degree)
        beta = NaiveSolverOLS(z, data, data_n, x_n, y_n)
        for j in range (0,degree+1):
            betaBundle[j][i] = beta[j]
    for i in range (0,degree+1):
        print ('Confindent interval for beta_{0} : [ {1} ; {2} ]' .format(i, Mean(betaBundle[i][:]) - 2*np.std(betaBundle[i][:]),  Mean(betaBundle[i][:]) + 2*np.std(betaBundle[i][:])))


#Number of grid points in one dim
N = 500

# Poly degree
degree = 4

# Interval
Experiments = 10

#Define interval for the arguments of the Franke function
x_exact = np.arange(0, 1, 0.05)
y_exact = np.arange(0, 1, 0.05)

z, data, data_n, x_n, y_n = SetUpData(N,degree)
print("########################################")
print("Scikit OSL: ")
ScikitSolverOLS(data, z, degree)
print(' ============== Scikit Lasso ============')
ScikitSolverLasso(data, z)
print(' ============== Scikit Ridge ============')
ScikitSolverRidge(data, z)
print(' ============== Naive Ridge ============') 
NaiveSolverRidge(data, z, x_n, y_n, data_n, degree)
print(" ============== Naive OSL ============= ")
NaiveSolverOLS(z, data, data_n, x_n, y_n)
print(' =============== Naive OLS ConfidentIntervalBeta ==============')  
ConfidentIntervalBeta(Experiments, N, degree)
