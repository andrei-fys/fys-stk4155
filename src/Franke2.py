from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from sklearn.linear_model import LinearRegression, RidgeCV, Lasso, Ridge, LassoCV 
from sklearn.metrics import mean_squared_error, r2_score 
import sklearn.preprocessing
import copy as cp
from functions import bias2, mse, ridge_regression_variance, R2
import sklearn.model_selection 
import sys
from tqdm import tqdm
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

''' SCI-KIT implementation'''

def OLS_SK(degree, X, x, y):
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
    print ('Variance  for OSL ', z_variance)
    print ('R2, Bias, MSE error for OSL ',r2,bias,mse_error)

    '''Make plot

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    print ('Shape z_n ', np.shape(zpredict))
    #print ('Shape z_n ', np.shape(x))
    zpredict = zpredict.reshape(1000,1000)
    #print ('Shape z_n ', np.shape(z_n))
    #print (np.shape(x), np.shape(y))
    surf = ax.plot_surface(x, y, zpredict, cmap=cm.coolwarm,
    linewidth=0, antialiased=False)
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    ax.text2D(0.05, 0.95, "The Franke function fitted", transform=ax.transAxes, color= 'blue')
    xLabel = ax.set_xlabel('\nx', linespacing=3.2)
    yLabel = ax.set_ylabel('\ny', linespacing=3.1)
    zLabel = ax.set_zlabel('\n f(x,y)', linespacing=0.5)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.draw()
    #plt.savefig("Franke_fitted_OLS.pdf")
    #plt.savefig("Franke_fitted_Ridge.pdf")
    #plt.savefig("Franke_fitted_Lasso.pdf")
    '''



def Rigde_SK(degree, X, alpha, x, y):
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
    print ('Variance', z_variance)

    '''Make plot

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    #print ('Shape z_n ', np.shape(zpredict))
    #print ('Shape z_n ', np.shape(x))
    zpredict = zpredict.reshape(1000,1000)
    #print ('Shape z_n ', np.shape(z_n))
    #print (np.shape(x), np.shape(y))
    surf = ax.plot_surface(x, y, zpredict, cmap=cm.coolwarm,
    linewidth=0, antialiased=False)
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    ax.text2D(0.05, 0.95, "The Franke function fitted", transform=ax.transAxes, color= 'blue')
    xLabel = ax.set_xlabel('\nx', linespacing=3.2)
    yLabel = ax.set_ylabel('\ny', linespacing=3.1)
    zLabel = ax.set_zlabel('\n f(x,y)', linespacing=0.5)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.draw()
    #plt.savefig("Franke_fitted_OLS.pdf")
    #plt.savefig("Franke_fitted_Ridge.pdf")
    #plt.savefig("Franke_fitted_Lasso.pdf")
    '''

    
def Lasso_SK(degree, X, alpha,x,y): 
    #alphas = np.arange(0.2, 0.6, 0.001) 
    #lasso = LassoCV(alphas=alphas, fit_intercept=False )
    lasso = Lasso(alpha=alpha, fit_intercept=False)
    z = FrankeFunction(x,y)
    lasso.fit(X, z.ravel())
    zpredict = lasso.predict(X)
    r2 = r2_score(z.ravel(), zpredict)
    R2 = lasso.score(X, z.ravel())
    bias = bias2(z.ravel(), zpredict)
    mse_error = mse(z.ravel(), zpredict)
    #print (r2,bias,mse_error)
    N, P = X.shape
    z_variance = np.sum((z.ravel() - zpredict)**2) / (N - P - 1)
    beta_variance = ridge_regression_variance(
            X, z_variance, alpha)
    #print ('Variance in beta for Lasso ', beta_variance)
    print ('R2, Bias, MSE error for Lasso ',r2, bias, mse_error)
    print ('Alternative r2 ',R2)
    print ('Variance', z_variance)

    '''Make plot

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    #print ('Shape z_n ', np.shape(zpredict))
    #print ('Shape z_n ', np.shape(x))
    zpredict = zpredict.reshape(1000,1000)
    #print ('Shape z_n ', np.shape(z_n))
    #print (np.shape(x), np.shape(y))
    surf = ax.plot_surface(x, y, zpredict, cmap=cm.coolwarm,
    linewidth=0, antialiased=False)
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    ax.text2D(0.05, 0.95, "The Franke function fitted", transform=ax.transAxes, color= 'blue')
    xLabel = ax.set_xlabel('\nx', linespacing=3.2)
    yLabel = ax.set_ylabel('\ny', linespacing=3.1)
    zLabel = ax.set_zlabel('\n f(x,y)', linespacing=0.5)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.draw()
    #plt.savefig("Franke_fitted_OLS.pdf")
    #plt.savefig("Franke_fitted_Ridge.pdf")
    plt.savefig("Franke_fitted_Lasso.pdf")
    '''


def OLS_CV_SKI(degree, x, y):
    """Scikit Learn method for cross validation."""
    z = FrankeFunction(x,y)
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
        np.c_[x.ravel(), y.ravel()], z.ravel(),
        test_size=0.2, shuffle=True)

    kf = sklearn.model_selection.KFold(n_splits=5)

    X_test = np.c_[np.ones(x_test.shape), x_test, x_test*x_test]
    X_train = np.c_[np.ones(x_train.shape), x_train, x_train*x_train]

    y_predicted = []
    beta_ = []

    for train_index, test_index in tqdm(kf.split(X_train), desc=None):

        kX_train, kX_test = X_train[train_index], X_train[test_index]
        kY_train, kY_test = y_train[train_index], y_train[test_index]
        kf_reg = LinearRegression(fit_intercept=False)
        kf_reg.fit(kX_train, kY_train)
        y_predicted.append(kf_reg.predict(X_test))
        beta_.append(kf_reg.coef_)
    y_predicted = np.asarray(y_predicted)

    # Mean Square Error
    _mse = (y_test - y_predicted)**2
    MSE = np.mean(np.mean(_mse, axis=0, keepdims=True))

    # Bias
    _mean_pred = np.mean(y_predicted, axis=0, keepdims=True)
    bias = np.mean((y_test - _mean_pred)**2)

    # R^2 
    R_2 = np.mean(R2(y_test, y_predicted, axis=0))

    # Variance
    var = np.mean(np.var(y_predicted, axis=0, keepdims=True))

    beta_variance = np.asarray(beta_).var(axis=0)
    beta_ = np.asarray(beta_).mean(axis=0)
    print ('Sci-Kit k fold for OLS results: ')
    print ('MSE',MSE)
    print ('R_2', R_2)
    print ('Beta coef variance ', beta_variance)
    print ('Bias2', bias)
    print ('Variance', var)


def Ridge_CV_SKI(degree, x, y, alpha):
    """Scikit Learn method for cross validation."""
    z = FrankeFunction(x,y)
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
        np.c_[x.ravel(), y.ravel()], z.ravel(),
        test_size=0.2, shuffle=True)

    kf = sklearn.model_selection.KFold(n_splits=5)

    X_test = np.c_[np.ones(x_test.shape), x_test, x_test*x_test]
    X_train = np.c_[np.ones(x_train.shape), x_train, x_train*x_train]

    y_predicted = []
    beta_ = []

    for train_index, test_index in tqdm(kf.split(X_train), desc=None):


        kX_train, kX_test = X_train[train_index], X_train[test_index]
        kY_train, kY_test = y_train[train_index], y_train[test_index]
        kf_reg = Ridge(alpha=alpha, solver="lsqr", fit_intercept=False) 
        kf_reg.fit(kX_train, kY_train)
        y_predicted.append(kf_reg.predict(X_test))
        beta_.append(kf_reg.coef_)
    y_predicted = np.asarray(y_predicted)

    # Mean Square Error
    _mse = (y_test - y_predicted)**2
    MSE = np.mean(np.mean(_mse, axis=0, keepdims=True))

    # Bias
    _mean_pred = np.mean(y_predicted, axis=0, keepdims=True)
    bias = np.mean((y_test - _mean_pred)**2)

    # R^2 
    R_2 = np.mean(R2(y_test, y_predicted, axis=0))

    # Variance
    var = np.mean(np.var(y_predicted, axis=0, keepdims=True))

    beta_variance = np.asarray(beta_).var(axis=0)
    beta_ = np.asarray(beta_).mean(axis=0)
    print ('Sci-Kit k fold for Ridge results: ')
    print ('MSE',MSE)
    print ('R_2', R_2)
    print ('Beta coef variance ', beta_variance)
    print ('Bias2', bias)
    print ('Variance', var)



def Lasso_CV_SKI(degree, x, y, alpha):
    """Scikit Learn method for cross validation."""
    z = FrankeFunction(x,y)
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
        np.c_[x.ravel(), y.ravel()], z.ravel(),
        test_size=0.1, shuffle=True)

    kf = sklearn.model_selection.KFold(n_splits=5)

    X_test = np.c_[np.ones(x_test.shape), x_test, x_test*x_test]
    X_train = np.c_[np.ones(x_train.shape), x_train, x_train*x_train]

    y_predicted = []
    beta_ = []

    for train_index, test_index in tqdm(kf.split(X_train), desc=None):

        kX_train, kX_test = X_train[train_index], X_train[test_index]
        kY_train, kY_test = y_train[train_index], y_train[test_index]
        kf_reg = Lasso(alpha=alpha, fit_intercept=False)
        kf_reg.fit(kX_train, kY_train)
        y_predicted.append(kf_reg.predict(X_test))
        beta_.append(kf_reg.coef_)
    y_predicted = np.asarray(y_predicted)

    # Mean Square Error
    _mse = (y_test - y_predicted)**2
    MSE = np.mean(np.mean(_mse, axis=0, keepdims=True))

    # Bias
    _mean_pred = np.mean(y_predicted, axis=0, keepdims=True)
    bias = np.mean((y_test - _mean_pred)**2)
    #print (np.shape(y_predicted))
   
    
    # R^2 
    R_2 = np.mean(R2(y_test, y_predicted, axis=0))

    # Variance
    var = np.mean(np.var(y_predicted, axis=0, keepdims=True))

    beta_variance = np.asarray(beta_).var(axis=0)
    beta_ = np.asarray(beta_).mean(axis=0)
    print ('Sci-Kit k fold for Lasso results: ')
    print ('MSE',MSE)
    print ('R_2', R_2)
    #print ('Beta coef variance ', beta_variance)
    print ('Bias2', bias)
    print ('Variance', var)


''' Manual implementation'''



if __name__ == '__main__':

    N=1000
    degree = int(sys.argv[1])
    alpha = float(sys.argv[2])
    #noise = sys.argv[2]
    #degree = 5
    x,y = SetUpGrid(N);
    z = FrankeFunction(x,y)
    X = SetUpDesignMatrix(degree, x,y)
    #print (np.size(X))
    #OLS_SK(degree, X, x, y)
    #alpha = float(sys.argv[2])
    #alpha = 0.2
    #Rigde_SK(degree, X, alpha, x, y)
    Lasso_SK(degree, X, alpha, x, y)
    #OLS_CV_SKI(degree, x, y)
    #Ridge_CV_SKI(degree, x, y, alpha)
    Lasso_CV_SKI(degree, x, y, alpha)
    
