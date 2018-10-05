from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
#np.set_printoptions(threshold=np.nan)


def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def SetUpDesignMat(x,y,N):
    "Set up matrix for the polynomial approximation	"
    assert len(x)==len(y)
    # Degree 1
    degree_one = x + y
    # Degree 2
    degree_two = x**2 + 2*x*y + y**2
    #Degree 3
    degree_three = x**3 + 3*x**2*y + 3*y**2*x + y**3
    # Degree 4
    degree_four = x**4 + 4*x**3*y + 6*x**2*y**2 + 4*y**3*x + y**4
    # Degree 5
    degree_five = x**5 + 5*x**4*y + 10*x**3*y**2 + 10*x**2*y**3 + 5*y**4*x + y**5
    return  np.hstack([np.ones((N*N,1)), degree_one, degree_two, degree_three, degree_four, degree_five])

def SetUpGrid(N):    
    x = np.sort(np.random.uniform(0,1,N))
    y = np.sort(np.random.uniform(0,1,N))
    mu, sigma = 0.5, 0.28
    x += np.random.normal(mu, sigma, N)
    y += np.random.normal(mu, sigma, N)
    x = np.sort(x)
    y = np.sort(y)
    x, y = np.meshgrid(x,y)
    return x, y
	

def OSL(N):
    x, y = SetUpGrid(N)
    z = FrankeFunction(x,y)
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    z = z.reshape(N*N,1)
    data = SetUpDesignMat(x,y,N)
    LinearReg = LinearRegression()
    LinearReg.fit(data,z)
    x_n, y_n = SetUpGrid(N)
    x_n = x_n.reshape(-1, 1)
    y_n = y_n.reshape(-1, 1)
    data_n = SetUpDesignMat(x_n,y_n,N)
    z_n = LinearReg.predict(data_n)
    R2 = LinearReg.score(data_n, z.reshape(-1, 1))
    MSE = mean_squared_error(z.reshape(-1, 1), z_n)
    print(R2)
    print(MSE)  
    print('Coefficient beta : \n', LinearReg.coef_)






#fig = plt.figure()
#ax = fig.gca(projection='3d')
# Make data.
#x_exact = np.arange(0, 1, 0.05)
#y_exact = np.arange(0, 1, 0.05)
#N = 500

#z = FrankeFunction(x_exact, y_exacti) 
## Plot the surface.
#surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
#linewidth=0, antialiased=False)
## Customize the z axis.
#ax.set_zlim(-0.10, 1.40)
#ax.zaxis.set_major_locator(LinearLocator(10))
#ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
## Add a color bar which maps values to colors.
#fig.colorbar(surf, shrink=0.5, aspect=5)
#plt.show()
if __name__ == '__main__':
    OSL(20)
