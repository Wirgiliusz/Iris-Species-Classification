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
# Wczytanie danych
iris = datasets.load_iris()
X = iris['data']
y = iris['target']

# Ustawienia do rysowania
h = 0.02
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# - - - - - KN - - - - - #
# - - - dla Sepal - - - #
Xs = X[:,0:2] # sepal length/width

# Stworzenie klasyfikatorow
# n_neighbors - liczba najblizszych sasiadow
knn1 = KNeighborsClassifier(n_neighbors=1)  
knn10 = KNeighborsClassifier(n_neighbors=10)
knn30 = KNeighborsClassifier(n_neighbors=30)
# Uczenie klasyfikatorow
knn1.fit(Xs, y)
knn10.fit(Xs, y)
knn30.fit(Xs, y)

# Ustawienia dla wykresu
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

# Tworzenie plotow
fig1 = plt.figure(1)
sub1 = fig1.add_subplot(231)
sub10 = fig1.add_subplot(232)
sub30 = fig1.add_subplot(233)

# rysowanie granic
sub1.pcolormesh(xx, yy, Z1, cmap=cmap_light)
sub10.pcolormesh(xx, yy, Z10, cmap=cmap_light)
sub30.pcolormesh(xx, yy, Z30, cmap=cmap_light)

# rysowanie punktow
sub1.scatter(Xs[:,0], Xs[:,1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
sub10.scatter(Xs[:,0], Xs[:,1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
sub30.scatter(Xs[:,0], Xs[:,1], c=y, cmap=cmap_bold, edgecolor='k', s=20)

# nazywanie wykresow
fig1.suptitle('Sepal Width(Length) and Petal Width(Length)\n for KNeighbors Classifier')
sub1.set_xlabel('Length [cm]')
sub1.set_ylabel('Width [cm]')
sub1.set_title('K=1')
sub10.set_xlabel('Length [cm]')
sub10.set_ylabel('Width [cm]')
sub10.set_title('[Sepal] K=10')
sub30.set_xlabel('Length [cm]')
sub30.set_ylabel('Width [cm]')
sub30.set_title('K=30')

# - - - dla Petal - - - #
Xp = X[:,2:4] # petal length/width

# Stworzenie klasyfikatorow
# n_neighbors - liczba najblizszych sasiadow
knn1 = KNeighborsClassifier(n_neighbors=1)  
knn10 = KNeighborsClassifier(n_neighbors=10)
knn30 = KNeighborsClassifier(n_neighbors=30)

# Uczenie klasyfikatorow
knn1.fit(Xp, y)
knn10.fit(Xp, y)
knn30.fit(Xp, y)

# Ustawienia dla wykresu
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

# tworzenie plotow
sub1 = fig1.add_subplot(234)
sub10 = fig1.add_subplot(235)
sub30 = fig1.add_subplot(236)

# rysowanie granic
sub1.pcolormesh(xx, yy, Z1, cmap=cmap_light)
sub10.pcolormesh(xx, yy, Z10, cmap=cmap_light)
sub30.pcolormesh(xx, yy, Z30, cmap=cmap_light)

#rysowanie punktow
sub1.scatter(Xp[:,0], Xp[:,1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
sub10.scatter(Xp[:,0], Xp[:,1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
sub30.scatter(Xp[:,0], Xp[:,1], c=y, cmap=cmap_bold, edgecolor='k', s=20)

#nazywanie wykresow
sub1.set_xlabel('Length [cm]')
sub1.set_ylabel('Width [cm]')
sub1.set_title('K=1')
sub10.set_xlabel('Length [cm]')
sub10.set_ylabel('Width [cm]')
sub10.set_title('[Petal] K=10')
sub30.set_xlabel('Length [cm]')
sub30.set_ylabel('Width [cm]')
sub30.set_title('K=30')


# - - - - - MLP - - - - - #
# - - - dla Sepal - - - #
Xs = X[:,0:2] # sepal length/width

# Stworzenie klasyfikatorow 
# hiddel_layer_sizes - liczba i wielkosc warstw ukrytych
# activation - funkcja aktywacji
# solver - algorytm uczenia
mlp1 = MLPClassifier(hidden_layer_sizes=(15,), activation="relu", solver='adam', max_iter=500)
mlp3 = MLPClassifier(hidden_layer_sizes=(15,15,15,), activation="relu", solver='adam', max_iter=500)
mlp5 = MLPClassifier(hidden_layer_sizes=(15,15,15,15,15,), activation="relu", solver='adam', max_iter=500)
mlp1.fit(Xs, y)
mlp3.fit(Xs, y)
mlp5.fit(Xs, y)

# Ustawienia wykresu
x_min, x_max = Xs[:,0].min()-.1, Xs[:,0].max()+.1
y_min, y_max = Xs[:,1].min()-.1, Xs[:,1].max()+.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Przewidywanie
Z1 = mlp1.predict(np.c_[xx.ravel(), yy.ravel()])    
Z3 = mlp3.predict(np.c_[xx.ravel(), yy.ravel()])
Z5 = mlp5.predict(np.c_[xx.ravel(), yy.ravel()])

Z1 = Z1.reshape(xx.shape)
Z3 = Z3.reshape(xx.shape)
Z5 = Z5.reshape(xx.shape)

# tworzenie plotow
fig2 = plt.figure(2)
sub1 = fig2.add_subplot(231)
sub3 = fig2.add_subplot(232)
sub5 = fig2.add_subplot(233)

# rysowanie granic
sub1.pcolormesh(xx, yy, Z1, cmap=cmap_light)
sub3.pcolormesh(xx, yy, Z3, cmap=cmap_light)
sub5.pcolormesh(xx, yy, Z5, cmap=cmap_light)

# rysowanie punktow
sub1.scatter(Xs[:,0], Xs[:,1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
sub3.scatter(Xs[:,0], Xs[:,1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
sub5.scatter(Xs[:,0], Xs[:,1], c=y, cmap=cmap_bold, edgecolor='k', s=20)

# nazywanie wykresow
fig2.suptitle('Sepal Width(Length) and Petal Width(Length)\n for M-L Perceptron')
sub1.set_xlabel('Length [cm]')
sub1.set_ylabel('Width [cm]')
sub1.set_title('N Layers = 1')
sub3.set_xlabel('Length [cm]')
sub3.set_ylabel('Width [cm]')
sub3.set_title('[Sepal] N Layers = 3')
sub5.set_xlabel('Length [cm]')
sub5.set_ylabel('Width [cm]')
sub5.set_title('N Layers = 5')

# - - - dla Petal - - - #
Xp = X[:,2:4] # petal length/width

# Stworzenie klasyfikatorow 
# hiddel_layer_sizes - liczba i wielkosc warstw ukrytych
# activation - funkcja aktywacji
# solver - algorytm uczenia
mlp1 = MLPClassifier(hidden_layer_sizes=(15,), activation="relu", solver='adam', max_iter=500)
mlp3 = MLPClassifier(hidden_layer_sizes=(15,15,15,), activation="relu", solver='adam', max_iter=500)
mlp5 = MLPClassifier(hidden_layer_sizes=(15,15,15,15,15,), activation="relu", solver='adam', max_iter=500)
mlp1.fit(Xp, y)
mlp3.fit(Xp, y)
mlp5.fit(Xp, y)

# Ustawienai wykresu
x_min, x_max = Xp[:,0].min()-.3, Xp[:,0].max()+.3
y_min, y_max = Xp[:,1].min()-.3, Xp[:,1].max()+.3
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# Przewidywanie

Z1 = mlp1.predict(np.c_[xx.ravel(), yy.ravel()])    
Z3 = mlp3.predict(np.c_[xx.ravel(), yy.ravel()])
Z5 = mlp5.predict(np.c_[xx.ravel(), yy.ravel()])

Z1 = Z1.reshape(xx.shape)
Z3 = Z3.reshape(xx.shape)
Z5 = Z5.reshape(xx.shape)

# tworzenie plotow
sub1 = fig2.add_subplot(234)
sub3 = fig2.add_subplot(235)
sub5 = fig2.add_subplot(236)

# rysowanie granic
sub1.pcolormesh(xx, yy, Z1, cmap=cmap_light)
sub3.pcolormesh(xx, yy, Z3, cmap=cmap_light)
sub5.pcolormesh(xx, yy, Z5, cmap=cmap_light)

# rysowanie punktow
sub1.scatter(Xp[:,0], Xp[:,1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
sub3.scatter(Xp[:,0], Xp[:,1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
sub5.scatter(Xp[:,0], Xp[:,1], c=y, cmap=cmap_bold, edgecolor='k', s=20)

# nazywanie wykresow
sub1.set_xlabel('Length [cm]')
sub1.set_ylabel('Width [cm]')
sub1.set_title('N Layers = 1')
sub3.set_xlabel('Length [cm]')
sub3.set_ylabel('Width [cm]')
sub3.set_title('[Petal] N Layers = 3')
sub5.set_xlabel('Length [cm]')
sub5.set_ylabel('Width [cm]')
sub5.set_title('N Layers = 5')

plt.show()
# - - - - - - - - - - - - - - #