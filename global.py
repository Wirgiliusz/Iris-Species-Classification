from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.neural_network import MLPClassifier  
from sklearn.neighbors import KNeighborsClassifier

from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt

# - - - - - WYKRESY GRANIC ROZNYCH ESTYMATOROW DLA ROZNYCH ARGUMENTOW - - - - - #
iris = datasets.load_iris()
X = iris['data']
y = iris['target']

# Ustawienia do rysowania
h = 0.02
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# - - - - - KN - - - - - #
# dla Sepal #
Xs = X[:,0:2] # sepal length/width
knn1 = KNeighborsClassifier(n_neighbors=1)
knn10 = KNeighborsClassifier(n_neighbors=10)
knn30 = KNeighborsClassifier(n_neighbors=30)
knn1.fit(Xs, y)
knn10.fit(Xs, y)
knn30.fit(Xs, y)

x_min, x_max = Xs[:,0].min()-.1, Xs[:,0].max()+.1
y_min, y_max = Xs[:,1].min()-.1, Xs[:,1].max()+.1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# Przewidywanie
Z1 = knn1.predict(np.c_[xx.ravel(), yy.ravel()])    
Z10 = knn10.predict(np.c_[xx.ravel(), yy.ravel()])
Z30 = knn30.predict(np.c_[xx.ravel(), yy.ravel()])

Z1 = Z1.reshape(xx.shape)
Z10 = Z10.reshape(xx.shape)
Z30 = Z30.reshape(xx.shape)

fig1 = plt.figure(1)
sub1 = fig1.add_subplot(231)
sub10 = fig1.add_subplot(232)
sub30 = fig1.add_subplot(233)

sub1.pcolormesh(xx, yy, Z1, cmap=cmap_light)
sub10.pcolormesh(xx, yy, Z10, cmap=cmap_light)
sub30.pcolormesh(xx, yy, Z30, cmap=cmap_light)

sub1.scatter(Xs[:,0], Xs[:,1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
sub10.scatter(Xs[:,0], Xs[:,1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
sub30.scatter(Xs[:,0], Xs[:,1], c=y, cmap=cmap_bold, edgecolor='k', s=20)

fig1.suptitle('Sepal Width(Length) and Petal Width(Length)\n for KNeighbors Classifier')
sub1.set_xlabel('Length [cm]')
sub1.set_ylabel('Width [cm]')
sub1.set_title('\nK=1')
sub10.set_xlabel('Length [cm]')
sub10.set_ylabel('Width [cm]')
sub10.set_title('[Sepal] K=10')
sub30.set_xlabel('Length [cm]')
sub30.set_ylabel('Width [cm]')
sub30.set_title('\nK=30')

# dla Petal #
Xp = X[:,2:4] # petal length/width
knn1 = KNeighborsClassifier(n_neighbors=1)
knn10 = KNeighborsClassifier(n_neighbors=10)
knn30 = KNeighborsClassifier(n_neighbors=30)
knn1.fit(Xp, y)
knn10.fit(Xp, y)
knn30.fit(Xp, y)

x_min, x_max = Xp[:,0].min()-.3, Xp[:,0].max()+.3
y_min, y_max = Xp[:,1].min()-.3, Xp[:,1].max()+.3

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# Przewidywanie
Z1 = knn1.predict(np.c_[xx.ravel(), yy.ravel()])    
Z10 = knn10.predict(np.c_[xx.ravel(), yy.ravel()])
Z30 = knn30.predict(np.c_[xx.ravel(), yy.ravel()])

Z1 = Z1.reshape(xx.shape)
Z10 = Z10.reshape(xx.shape)
Z30 = Z30.reshape(xx.shape)

sub1 = fig1.add_subplot(234)
sub10 = fig1.add_subplot(235)
sub30 = fig1.add_subplot(236)

sub1.pcolormesh(xx, yy, Z1, cmap=cmap_light)
sub10.pcolormesh(xx, yy, Z10, cmap=cmap_light)
sub30.pcolormesh(xx, yy, Z30, cmap=cmap_light)

sub1.scatter(Xp[:,0], Xp[:,1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
sub10.scatter(Xp[:,0], Xp[:,1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
sub30.scatter(Xp[:,0], Xp[:,1], c=y, cmap=cmap_bold, edgecolor='k', s=20)

sub1.set_xlabel('Length [cm]')
sub1.set_ylabel('Width [cm]')
sub1.set_title('K=1')
sub10.set_xlabel('Length [cm]')
sub10.set_ylabel('Width [cm]')
sub10.set_title('[Petal] K=10')
sub30.set_xlabel('Length [cm]')
sub30.set_ylabel('Width [cm]')
sub30.set_title('K=30')

plt.show()