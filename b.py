#!/usr/bin/env python
#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.pyplot as plt
#from matplotlib import cm
#from matplotlib.ticker import LinearLocator, FormatStrFormatter
import sys
import numpy as np
from random import random, seed
import sklearn.model_selection
from sklearn.linear_model import LinearRegression,RidgeCV,Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from Franke1 import FrankeFunction, SetUpDesignMat, SetUpGrid
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split

def ScikitSolverOLS(N, degree, noise, z, data, data_n):
    #z, data, data_n, x_n, y_n, x, y = SetUpData(N,degree,noise)
    clf5 = LinearRegression(fit_intercept=False)
    clf5.fit(data,z)
    z_n = clf5.predict(data_n)
    R2 = clf5.score(data_n, z)
    MSE = mean_squared_error(z, z_n)
    print('R2 SciKit', R2)
    print('MSE SciKit  ',MSE)
    print('Coefficient beta : \n', clf5.coef_)

    print ('!!!!!!!CV start here !!!!!!!')
    predict = cross_val_predict(clf5, data, z, cv=5)
    scores = cross_val_score(clf5, data, z, cv=5)
    print('scores : ', scores) 
    print ('R2 cross-valid ', r2_score(predict, z))
    print ('Scores ', scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print('MEAN ', scores.mean()) 
    print('STD ',scores.std()) 

def ScikitSolverRidge(N,degree,noise, z, data, data_n):
    #z, data, data_n, _, _, _, _ = SetUpData(N,degree,noise)
    #ridge=RidgeCV(alphas=[0.1,1.0,10.0])
    ridge=Ridge(alpha=0.1, fit_intercept=False)
    ridge.fit(data,z)
    R2 = ridge.score(data_n, z)
    print("Ridge Coefficient: ",ridge.coef_)
    print("Ridge Intercept: ", ridge.intercept_)
    
def ScikitSolverLasso(N,degree,noise, z, data, data_n):
    #z, data, data_n, _, _, _, _ = SetUpData(N,degree,noise)
    lasso=Lasso(alpha=0.2)
    lasso.fit(data,z)
    predl=lasso.predict(data)
    R2 = lasso.score(data_n, z)
    print('R2 SciKit', R2)
    print("Lasso Coefficient: ", lasso.coef_)
    print("Lasso Intercept: ", lasso.intercept_)

def NaiveSolverOLS(N,degree,noise, z, data, data_n, x_n, y_n):
    #z, data, data_n, x_n, y_n, _, _ = SetUpData(N,degree,noise)
    beta = np.linalg.inv(data.T.dot(data)).dot(data.T).dot(z)
    zpredict = data_n.dot(beta)
    #print('Coefficient beta naive: \n', beta.reshape(1,-1))
    beta = beta.reshape(-1,1)
    #print(' Beta0 ', beta[0])
    #print('Mean naive ', Mean(z))
    #print('R2 naive ', R2(z, zpredict))
    #print('MSE naive  ', MSE(z, zpredict))
    print(Mean(z), R2(z, zpredict), MSE(z, zpredict))
    return beta

def NaiveSolverRidge(N,degree,noise, z, data, data_n, x_n, y_n):
    #z, data, data_n, x_n, y_n, _, _ = SetUpData(N,degree,noise)
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

def SetUpData(N, degree,noise):
    x, y = SetUpGrid(N,noise)
    z = FrankeFunction(x,y)
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    z = z.reshape(N*N,1)
    data = SetUpDesignMat(x,y,N,degree)
    x_n, y_n = SetUpGrid(N,noise)
    x_n = x_n.reshape(-1, 1)
    y_n = y_n.reshape(-1, 1)
    data_n = SetUpDesignMat(x_n,y_n,N,degree)
    return z, data, data_n, x_n, y_n, x, y

def ConfidentIntervalBeta(Experiments, N, degree, noise):
    betaBundle = np.zeros(shape=(degree+1,Experiments))
    for i in range (0, Experiments):
        z, data, data_n, x_n, y_n, x, y = SetUpData(N,degree,noise)
        beta = NaiveSolverOLS(z, data, data_n, x_n, y_n)
        for j in range (0,degree+1):
            betaBundle[j][i] = beta[j]
    for i in range (0,degree+1):
        print ('Confindent interval for beta_{0} : [ {1} ; {2} ]' .format(i, Mean(betaBundle[i][:]) - 2*np.std(betaBundle[i][:]),  Mean(betaBundle[i][:]) + 2*np.std(betaBundle[i][:])))

'''
def kFoldCV(N,degree,noise):

    x,y = SetUpGrid(N,noise)
    z = FrankeFunction(x,y)
    zpredict = []
    XY_train, XY_test, z_train, z_test = sklearn.model_selection.train_test_split(np.c_[x.ravel(), y.ravel()], z.ravel(), test_size=0.2)
    KFold = sklearn.model_selection.KFold(n_splits=15)  
    for train, test in KFold.split(XY_train):
        xy_kf_train, xy_kf_test = XY_train[train], XY_train[test]
        z_kf_train, z_kf_test = z_train[train], z_train[test]
        kfLinReg = LinearRegression(fit_intercept=False)
        kfLinReg.fit(xy_kf_train, z_kf_train)
        zpredict.append(kfLinReg.predict(XY_test))

    zpredict = np.asarray(zpredict)

    # MSE
    mse_current = (z_test - zpredict)**2
    MSE_mean = np.mean(np.mean(mse_current, axis=0, keepdims=True))

    # Bias, (y - mean(y_approx))^2
    predict_mean = np.mean(zpredict, axis=0, keepdims=True)
    bias = np.mean((z_test - predict_mean)**2)

    #R2
    #R2 = 

    # Variance 
    var = np.mean(np.var(zpredict, axis=0, keepdims=True))
    print('MSE = ', MSE_mean)
    print('Bias^2 = ', bias)
    print('Variance  = ', var)
    print('|MSE - bias - variance| = ', abs(MSE_mean - bias - var))
'''
def ScikitSolverRidgeCV(N, degree, noise, z, data):
    #z, data, _, _, _, _, _ = SetUpData(N,degree,noise)
    X_train, X_test, y_train, y_test = train_test_split(data, z, test_size=0.2, random_state=0)
    ridge=Ridge(alpha=0.1, fit_intercept=False)
    ridge.fit(X_train,y_train)
    y_predict = ridge.predict(X_test)
    #R_2 = R2(y_test, y_predict)
    #print ('R2: ', R_2)
    print('R2: ',ridge.score(X_test, y_test))

def ScikitSolverLassoCV(N, degree, noise, z, data):
    #z, data, _, _, _, _, _ = SetUpData(N,degree,noise)
    X_train, X_test, y_train, y_test = train_test_split(data, z, test_size=0.2, random_state=0)
    lasso=Lasso(alpha=0.1, fit_intercept=False)
    lasso.fit(X_train,y_train)
    y_predict = lasso.predict(X_test)
    #R_2 = R2(y_test, y_predict)
    #print ('R2: ', R_2)
    print('R2: ', lasso.score(X_test, y_test))

def ScikitSolverOSLCV(N, degree, noise, z, data):
    #z, data, _, _, _, _, _ = SetUpData(N,degree,noise)
    X_train, X_test, y_train, y_test = train_test_split(data, z, test_size=0.2, random_state=0)
    OSL=LinearRegression( fit_intercept=False)
    OSL.fit(X_train,y_train)
    y_predict = OSL.predict(X_test)
    #R_2 = R2(y_test, y_predict)
    #print ('R2: ', R_2)
    print('R2: ', OSL.score(X_test, y_test))


#Number of grid points in one dim
N = 200

# Poly degree
#degree = 4
degree = int(sys.argv[1])
# Interval
Experiments = 10

# Noise
#noise = True
noise = sys.argv[2]

# Resampling
resampling = True

#Define interval for the arguments of the Franke function
x_exact = np.arange(0, 1, 0.05)
y_exact = np.arange(0, 1, 0.05)

z, data, data_n, x_n, y_n, x, y = SetUpData(N,degree,noise)

#print("########################################")
#kFoldCV(N,degree,noise)
#ScikitSolverRidge(N,degree,noise)
#print("########################################")
#NaiveSolverRidge(N,degree,noise)
#print("########################################")
#ScikitSolverRidgeCV(N, degree, noise)
print("############ OSL SciKit #########################")
ScikitSolverOLS(N, degree, noise, z, data, data_n)
print ("################## CV OSL ######################")
ScikitSolverOSLCV(N, degree, noise, z, data)

print ('=============== Ridge SciKit ===================')
ScikitSolverRidge(N, degree, noise, z, data, data_n)
print ('=============== Ridge CV =======================')
ScikitSolverRidgeCV(N, degree, noise, z, data)

print ('*************** Lasso Scikit *******************')
ScikitSolverLasso(N, degree, noise, z, data, data_n)
print ('*************** Lasso CV *******************')
ScikitSolverLassoCV(N, degree, noise, z, data)






