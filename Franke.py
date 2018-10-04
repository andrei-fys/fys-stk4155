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


fig = plt.figure()
ax = fig.gca(projection='3d')
# Make data.
x_exact = np.arange(0, 1, 0.05)
y_exact = np.arange(0, 1, 0.05)

x = np.sort(np.random.uniform(0,1,20))
y = np.sort(np.random.uniform(0,1,20))
x = x_exact
y = y_exact
#mu, sigma = 0.5, 0.28
#x += np.random.normal(mu, sigma, 20)
#y += np.random.normal(mu, sigma, 20)
x = np.sort(x)
y = np.sort(y)
x, y = np.meshgrid(x,y)
z = FrankeFunction(x,y)


#print(np.shape(y))

#data = np.c_[np.ones((20,1)), x, y, x**2, x*y, y**2, x**3, x**2*y, x*y**2, y**3, x**4, x**3*y, x**2*y**2, x*y**3,y**4, x**5, x**4*y, x**3*y**2, x**2*y**3,x*y**4, y**5]

x = x.reshape(-1, 1)
y = y.reshape(-1, 1)
z = z.reshape(400,1)

#print(np.shape(x))
data = np.hstack([np.ones((400,1)), x + y, x**2 + 2*x*y + y**2, x**3 + 3*x**2*y + 3*x*y**2 + y**3, x**4 + 4*x**3*y + 6*x**2*y**2 + 4*x*y**3 + y**4, x**5 + 5*x**4*y + 10*x**3*y**2 + 10*x**2*y**3 + 5*x*y**4 + y**5])
clf5 = LinearRegression()
clf5.fit(data,z)

x_n = np.sort(np.random.uniform(0,1,20))
y_n = np.sort(np.random.uniform(0,1,20))
x_n = x_exact
y_n = y_exact
#mu, sigma = 0.5, 0.28
#x += np.random.normal(mu, sigma, 20)
#y += np.random.normal(mu, sigma, 20)
x_n = np.sort(x_n)
y_n = np.sort(y_n)
x_n, y_n = np.meshgrid(x_n,y_n)
x_n = x_n.reshape(-1, 1)
y_n = y_n.reshape(-1, 1)
data_n = np.hstack([np.ones((400,1)), x_n + y_n, x_n**2 + 2*x_n*y_n + y_n**2, x_n**3 + 3*x_n**2*y_n + 3*x_n*y_n**2 + y_n**3, x_n**4 + 4*x_n**3*y_n + 6*x_n**2*y_n**2 + 4*x_n*y_n**3 + y_n**4, x_n**5 + 5*x_n**4*y_n + 10*x_n**3*y_n**2 + 10*x_n**2*y_n**3 + 5*x_n*y_n**4 + y_n**5])
z_n = clf5.predict(data_n)
R2 = clf5.score(data_n, z.reshape(-1, 1))
MSE = mean_squared_error(z.reshape(-1, 1), z_n)


print(R2)
print(MSE)

#linreg = LinearRegression()

#z = FrankeFunction(x, y) #+ np.random.randn(20, 20)

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
