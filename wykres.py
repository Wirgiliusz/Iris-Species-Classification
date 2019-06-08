from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.neural_network import MLPClassifier  
from sklearn.neighbors import KNeighborsClassifier

from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt

iris = datasets.load_iris()

iris = datasets.load_iris()
X = iris['data']
X = X[:,0:2]
y = iris['target']

# Ustawienia do rysowania
h = 0.02
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Sieci

clf = KNeighborsClassifier(n_neighbors=15)

clf.fit(X,y)
mlp.fit(X,y)
mlp2.fit(X,y)
mlp3.fit(X,y)

x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
y_min, y_max = X[:, 1].min()-1, X[:, 1].max()+1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z2 = mlp.predict(np.c_[xx.ravel(), yy.ravel()])
Z3 = mlp2.predict(np.c_[xx.ravel(), yy.ravel()])
Z4 = mlp3.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)
Z2 = Z2.reshape(xx.shape)
Z3 = Z3.reshape(xx.shape)
Z4 = Z4.reshape(xx.shape)
plt.figure(1)
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

plt.scatter(X[:,0], X[:,1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

plt.figure(2)
plt.pcolormesh(xx, yy, Z2, cmap=cmap_light)

plt.scatter(X[:,0], X[:,1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

plt.figure(3)
plt.pcolormesh(xx, yy, Z3, cmap=cmap_light)

plt.scatter(X[:,0], X[:,1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

plt.figure(4)
plt.pcolormesh(xx, yy, Z4, cmap=cmap_light)

plt.scatter(X[:,0], X[:,1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

plt.show()