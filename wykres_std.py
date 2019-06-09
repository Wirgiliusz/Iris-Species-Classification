from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.neural_network import MLPClassifier  

from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt


iris = datasets.load_iris()
X = iris['data']
y = iris['target']
print(iris["feature_names"])



# Podzial X po 2 cechy
X_sL = X[:,0]   # sepal length
X_sW = X[:,1]   # sepal width
X_sLW = X[:,0:2] # sepal length/width
X_pL = X[:,2]   # petal length
X_pW = X[:,3]   # petal width
X_pLW = X[:,2:4] # petal length/width

# Podzial obu na grupe treningowa i testowa
X_sLW_train, X_sLW_test, y_train, y_test = train_test_split(X_sLW, y, test_size=0.2) 
X_pLW_train, X_pLW_test, y_train, y_test = train_test_split(X_pLW, y, test_size=0.2)
X_sL_train, X_sL_test, y_train, y_test = train_test_split(X_sL, y, test_size=0.2)
X_sW_train, X_sW_test, y_train, y_test = train_test_split(X_sW, y, test_size=0.2)

# Normalizacja i standaryzacja danych
sc = StandardScaler()

sc.fit(X_sLW_train)     # Dla sepal
X_sLW_train = sc.transform(X_sLW_train)
X_sLW_test = sc.transform(X_sLW_test)

sc.fit(X_pLW_train)     # Dla petal
X_pLW_train = sc.transform(X_pLW_train)
X_pLW_test = sc.transform(X_pLW_test)


# Ustawienia do rysowania
h = 0.02
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])


# Sieci
mlp = MLPClassifier(hidden_layer_sizes=(10, 10,), activation="relu", solver='adam', max_iter=1000)
mlp.fit(X_sLW_train, y_train)

x_min, x_max = X_sLW_train.min()-1, X_sLW_train.max()+1
y_min, y_max = X_sW_train.min()-1, X_sW_train.max()+1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = mlp.predict(X_sLW_test)

Z = Z.reshape(xx.shape)
plt.figure(1)
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)


plt.scatter(X_sL_test, X_sW_test, c=y_test, cmap=cmap_bold, edgecolor='k', s=20)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

plt.show()