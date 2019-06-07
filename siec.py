from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.neural_network import MLPClassifier  

import numpy as np
import matplotlib.pyplot as plt


iris = datasets.load_iris()

X = iris['data']
y = iris['target']

print(iris.keys())
print(iris["target_names"])
print(iris["feature_names"])

s_l = []
p_l = []
s_w = []
p_w = []
kolor = []
nazwa = ['Setosa', 'Versicolor', 'Virginica', '']
for i in range(0, 150):
    s_l.append(X[i][0])
    p_l.append(X[i][2])
    s_w.append(X[i][1])
    p_w.append(X[i][3])
    if(y[i] == 0):
        kolor.append('ro')
    elif(y[i] == 1):
        kolor.append('go')
    elif(y[i] == 2):
        kolor.append('bo')

for i in range(0, 150):
    if(i==0):
        n = nazwa[0]
    elif(i==50):
        n = nazwa[1]
    elif(i==100):
        n = nazwa[2]
    else:
        n = nazwa[3]

    plt.subplot(221)
    plt.plot(s_l[i], p_l[i], color=kolor[i][0], marker=kolor[i][1], label=n)
    plt.subplot(222)
    plt.plot(s_w[i], p_w[i], color=kolor[i][0], marker=kolor[i][1], label=n)
    plt.subplot(223)
    plt.plot(s_l[i], s_w[i], color=kolor[i][0], marker=kolor[i][1], label=n)
    plt.subplot(224)
    plt.plot(p_w[i], p_l[i], color=kolor[i][0], marker=kolor[i][1], label=n)

plt.subplot(221)
plt.title("Petal(Sepal) length")
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend()

plt.subplot(222)
plt.title("Petal(Sepal) width")
plt.xlabel('Sepal width [cm]')
plt.ylabel('Petal width [cm]')
plt.legend()

plt.subplot(223)
plt.title("Sepal_len(Sepal_wid)")
plt.xlabel('Sepal width [cm]')
plt.ylabel('Sepal length [cm]')
plt.legend()

plt.subplot(224)
plt.title("Petal_len(Petal_wid)")
plt.xlabel('Petal width [cm]')
plt.ylabel('Petal length [cm]')
plt.legend()

#plt.show()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

sc = StandardScaler()
sc.fit(X_train)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


#ppn = Perceptron(max_iter=40, eta0=0.1, random_state=0)
#ppn.fit(X_train_std, y_train)
mlp = MLPClassifier(hidden_layer_sizes=(10, 10), activation="relu", solver='adam', max_iter=1000)
mlp.fit(X_train, y_train)  


#y_pred = ppn.predict(X_test_std)
y_pred = mlp.predict(X_test)


print(y_pred)
print(y_test)
print(len(y_test))

print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))