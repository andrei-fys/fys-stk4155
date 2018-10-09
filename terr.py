#import scipy.misc.imread
import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Load the terrain
terrain1 = imread('n11_w009_1arc_v3.tif')
# Show the terrain
#plt.figure()
#plt.title('Terrain over Norway 1')
#plt.imshow(terrain1, cmap='gray')
#plt.xlabel('X')
#plt.ylabel('Y')
#plt.show()

print(terrain1.shape)

# Extract a smaller patch of the terrain
#row_start = 1550  # 1950
#row_end = 1650  # 2050
#col_start = 600  # 1200
#col_end = 850  # 1450

row_start = 2000 
row_end = 2500 
col_start = 1000
col_end = 1500

terrain_patch = terrain1[row_start:row_end, col_start:col_end]
terrain_patch = terrain_patch/np.amax(terrain_patch)
#plt.figure()
#plt.imshow(terrain_patch, cmap='gray')
#plt.xlabel('X')
#plt.ylabel('Y')
#plt.show()
#

#print(terrain_patch.shape)


#
## Normalizes
#terrain_patch = terrain_patch/np.amax(terrain_patch)
#
# Sets up X,Y,Z 
M, N = terrain_patch.shape

#print (row_end-row_start, M)
#print (col_end-col_start, N)

# ax_rows = np.arange(row_start, row_start+M, 1)
# ax_cols = np.arange(col_start, col_start+N, 1)

ax_rows = np.arange(M)
ax_cols = np.arange(N)

[x, y] = np.meshgrid(ax_cols, ax_rows)
z = terrain_patch

M1, N1 = terrain_patch.shape
ax_rows = np.arange(M1)
ax_cols = np.arange(N1)
[X, Y] = np.meshgrid(ax_cols, ax_rows)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, terrain_patch, cmap='gray', linewidth=0)
plt.show()
