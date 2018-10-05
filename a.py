from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from Franke import FrankeFunction, SetUpDesignMat, SetUpGrid

def ScikitSolverOLS(data, z):
    clf5 = LinearRegression()
    clf5.fit(data,z)
    x_n, y_n = SetUpGrid(N)
    x_n = x_n.reshape(-1, 1)
    y_n = y_n.reshape(-1, 1)
    data_n = SetUpDesignMat(x_n,y_n,N)
    z_n = clf5.predict(data_n)
    R2 = clf5.score(data_n, z.reshape(-1, 1))
    MSE = mean_squared_error(z.reshape(-1, 1), z_n)
    print('R2 OLS SK', R2)
    print('MSE OLS SK  ',MSE)
    print('Coefficient beta : \n', clf5.coef_)

def ScikitSolverRidge(data, z):
    clf5 = LinearRegression()
    clf5.fit(data,z)
    x_n, y_n = SetUpGrid(N)
    x_n = x_n.reshape(-1, 1)
    y_n = y_n.reshape(-1, 1)
    data_n = SetUpDesignMat(x_n,y_n,N)
    z_n = clf5.predict(data_n)
    R2 = clf5.score(data_n, z.reshape(-1, 1))
    MSE = mean_squared_error(z.reshape(-1, 1), z_n)
    print('R2 RIDGE SK ', R2)
    print('MSE RIDGE SK ', MSE)
    print('Coefficient beta : \n', clf5.coef_)

#def ScikitSolverLasso(data, z):
#    clf5 = LinearRegression()
#    clf5.fit(data,z)
#    x_n, y_n = SetUpGrid(N)
#    x_n = x_n.reshape(-1, 1)
#    y_n = y_n.reshape(-1, 1)
#    data_n = SetUpDesignMat(x_n,y_n,N)
#    z_n = clf5.predict(data_n)
#    R2 = clf5.score(data_n, z.reshape(-1, 1))
#    MSE = mean_squared_error(z.reshape(-1, 1), z_n)
#    print(R2)
#    print(MSE)
#    print('Coefficient beta : \n', clf5.coef_)



def NaiveSolver(data, z):	
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

ScikitSolverOLS(data, z)
#ScikitSolverRidge(data, z)
#ScikitSolverLasso(data, z)
NaiveSolver(data, z)
