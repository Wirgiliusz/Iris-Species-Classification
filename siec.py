from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier  
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV

import numpy as np
import matplotlib.pyplot as plt


iris = datasets.load_iris()
X = iris['data']
y = iris['target']

#print(iris.keys())
#print(iris["target_names"])
#print(iris["feature_names"])

# - - - - - WYKRESY - - - - - #
'''
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


fig1 = plt.figure(1)
sub1 = fig1.add_subplot(221)
sub2 = fig1.add_subplot(222)
sub3 = fig1.add_subplot(223)
sub4 = fig1.add_subplot(224)
for i in range(0, 150):
    if(i==0):
        n = nazwa[0]
    elif(i==50):
        n = nazwa[1]
    elif(i==100):
        n = nazwa[2]
    else:
        n = nazwa[3]

    sub1.plot(s_l[i], p_l[i], color=kolor[i][0], marker=kolor[i][1], label=n)
    sub2.plot(s_w[i], p_w[i], color=kolor[i][0], marker=kolor[i][1], label=n)
    sub3.plot(s_l[i], s_w[i], color=kolor[i][0], marker=kolor[i][1], label=n)
    sub4.plot(p_w[i], p_l[i], color=kolor[i][0], marker=kolor[i][1], label=n)


sub1.set_title("Petal(Sepal) length")
sub1.set_xlabel('Sepal length [cm]')
sub1.set_ylabel('Petal length [cm]')
sub1.legend()

sub2.set_title("Petal(Sepal) width")
sub2.set_xlabel('Sepal width [cm]')
sub2.set_ylabel('Petal width [cm]')
sub2.legend()

sub3.set_title("Sepal_len(Sepal_wid)")
sub3.set_xlabel('Sepal width [cm]')
sub3.set_ylabel('Sepal length [cm]')
sub3.legend()

sub4.set_title("Petal_len(Petal_wid)")
sub4.set_xlabel('Petal width [cm]')
sub4.set_ylabel('Petal length [cm]')
sub4.legend()

#plt.show()
'''
# - - - - - - - - - - - - - - - #

# Podzial danych na grupe uczenia i testowania
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Normalizacja i standaryzacja danych
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


# - Perceptron - #
ppn = Perceptron(max_iter=40, eta0=0.1, random_state=0)
ppn.fit(X_train_std, y_train)
y_pred_ppn = ppn.predict(X_test_std)

# - Wielowarstwowy perceptron - #
mlp = MLPClassifier(hidden_layer_sizes=(10, 10), activation="relu", solver='adam', max_iter=1000)
mlp.fit(X_train_std, y_train)  
y_pred_mlp = mlp.predict(X_test_std)

# - K najblizszych sasiadow - #
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_std, y_train)
y_pred_knn = knn.predict(X_test_std)

plik = open('Wyniki.txt', 'w')
plik.write("Porownanie rzeczywistych wynikow z przewidywaniami:\n")
plik.write(str(y_test) + ' - rzeczywiste\n')
plik.write(str(y_pred_ppn) + ' - perceptron\n')
plik.write(str(y_pred_mlp) + ' - wielowarstwowy perceptron\n')
plik.write(str(y_pred_knn) + ' - K najblizszych sasiadow\n\n')
'''
print("Rzeczywiste wyniki:")
print(y_test)
print("Przewidywania peceptrona:")
print(y_pred_ppn)
print("Przewidywania wielowarstwowego perceptrona:")
print(y_pred_mlp)
print("Przewidywania K najblizszych sasiadow:")
print(y_pred_knn)
'''
plik.write('Dokladnosc perceptrona: ' + str(round(accuracy_score(y_test, y_pred_ppn)*100,2)) + "%\n")
plik.write('Dokladnosc wielowarstwowego perceptrona: ' + str(round(accuracy_score(y_test, y_pred_mlp)*100,2)) + "%\n")
plik.write('Dokladnosc K najblizszych sasiadow: ' + str(round(accuracy_score(y_test, y_pred_knn)*100,2)) + "%\n\n")
'''
print('Dokladnosc perceptrona: %.2f' % (accuracy_score(y_test, y_pred_ppn)*100), "%")
print('Dokladnosc wielowarstwowego perceptrona: %.2f' % (accuracy_score(y_test, y_pred_mlp)*100), "%")
print('Dokladnosc K najblizszych sasiadow: %.2f' % (accuracy_score(y_test, y_pred_knn)*100), "%")
'''  
# - Wielowarstwowy perceptron - #
# -      test wielokrotny     - #
dokladnosc1 = 0
dokladnosc2 = 0
dokladnosc3 = 0
dokladnosc4 = 0
for i in range(0,100):
    mlp1 = MLPClassifier(hidden_layer_sizes=(10, 10), activation="relu", solver='adam', max_iter=500)
    mlp2 = MLPClassifier(hidden_layer_sizes=(10, 10), activation="tanh", solver='adam', max_iter=500)
    mlp3 = MLPClassifier(hidden_layer_sizes=(10, 10), activation="relu", solver='sgd', max_iter=500)
    mlp4 = MLPClassifier(hidden_layer_sizes=(10, 10), activation="tanh", solver='sgd', max_iter=500)
    mlp1.fit(X_train_std, y_train)  
    mlp2.fit(X_train_std, y_train)  
    mlp3.fit(X_train_std, y_train)  
    mlp4.fit(X_train_std, y_train)  
    y_pred_mlp1 = mlp1.predict(X_test_std)
    y_pred_mlp2 = mlp2.predict(X_test_std)
    y_pred_mlp3 = mlp3.predict(X_test_std)
    y_pred_mlp4 = mlp4.predict(X_test_std)
    dokladnosc1 += accuracy_score(y_test, y_pred_mlp1)
    dokladnosc2 += accuracy_score(y_test, y_pred_mlp2)
    dokladnosc3 += accuracy_score(y_test, y_pred_mlp3)
    dokladnosc4 += accuracy_score(y_test, y_pred_mlp4)

dokladnosc_srednia1 = dokladnosc1/100
dokladnosc_srednia2 = dokladnosc2/100
dokladnosc_srednia3 = dokladnosc3/100
dokladnosc_srednia4 = dokladnosc4/100
plik.write('MLP (relu, adam): ' + str(round(dokladnosc_srednia1*100,2)) + "%\n")
plik.write('MLP (tanh, adam): ' + str(round(dokladnosc_srednia2*100,2)) + "%\n")
plik.write('MLP (relu, sgd): ' + str(round(dokladnosc_srednia3*100,2)) + "%\n")
plik.write('MLP (tanh, sgd): ' + str(round(dokladnosc_srednia4*100,2)) + "%\n\n")
'''
print('MLP (relu, adam): %.2f' % (dokladnosc_srednia1*100), "%")
print('MLP (tanh, adam): %.2f' % (dokladnosc_srednia2*100), "%")
print('MLP (relu, sgd): %.2f' % (dokladnosc_srednia3*100), "%")
print('MLP (tanh, sgd): %.2f' % (dokladnosc_srednia4*100), "%")
'''

dokladnosc1 = 0
dokladnosc2 = 0
dokladnosc3 = 0
dokladnosc4 = 0
for i in range(0,100):
    mlp1 = MLPClassifier(hidden_layer_sizes=(10,), activation="relu", solver='adam', max_iter=500)
    mlp2 = MLPClassifier(hidden_layer_sizes=(10, 10,), activation="relu", solver='adam', max_iter=500)
    mlp3 = MLPClassifier(hidden_layer_sizes=(10, 10, 10,), activation="relu", solver='adam', max_iter=500)
    mlp4 = MLPClassifier(hidden_layer_sizes=(10, 10, 10, 10,), activation="relu", solver='adam', max_iter=500)
    mlp1.fit(X_train_std, y_train)  
    mlp2.fit(X_train_std, y_train)  
    mlp3.fit(X_train_std, y_train)  
    mlp4.fit(X_train_std, y_train)  
    y_pred_mlp1 = mlp1.predict(X_test_std)
    y_pred_mlp2 = mlp2.predict(X_test_std)
    y_pred_mlp3 = mlp3.predict(X_test_std)
    y_pred_mlp4 = mlp4.predict(X_test_std)
    dokladnosc1 += accuracy_score(y_test, y_pred_mlp1)
    dokladnosc2 += accuracy_score(y_test, y_pred_mlp2)
    dokladnosc3 += accuracy_score(y_test, y_pred_mlp3)
    dokladnosc4 += accuracy_score(y_test, y_pred_mlp4)

dokladnosc_srednia1 = dokladnosc1/100
dokladnosc_srednia2 = dokladnosc2/100
dokladnosc_srednia3 = dokladnosc3/100
dokladnosc_srednia4 = dokladnosc4/100
plik.write('MLP [relu, adam] (10,): ' + str(round(dokladnosc_srednia1*100,2)) + "%\n")
plik.write('MLP [relu, adam] (10, 10,): ' + str(round(dokladnosc_srednia2*100,2)) + "%\n")
plik.write('MLP [relu, adam] (10, 10, 10,): ' + str(round(dokladnosc_srednia3*100,2)) + "%\n")
plik.write('MLP [relu, adam] (10, 10, 10, 10,): ' + str(round(dokladnosc_srednia4*100,2)) + "%\n")
'''
print('MLP [relu, adam] (10,): %.2f' % (dokladnosc_srednia1*100), "%")
print('MLP [relu, adam] (10, 10,): %.2f' % (dokladnosc_srednia2*100), "%")
print('MLP [relu, adam] (10, 10, 10,): %.2f' % (dokladnosc_srednia3*100), "%")
print('MLP [relu, adam] (10, 10, 10, 10,): %.2f' % (dokladnosc_srednia4*100), "%")
'''

# - Szukanie najlepszych parametrow - #
'''
param_grid = [
    {
        'activation' : ['identity', 'logistic', 'tanh', 'relu'],
        'solver' : ['lbfgs', 'sgd', 'adam'],
        'hidden_layer_sizes': [
        (1,),(2,),(3,),(4,),(5,),(6,),(7,),(8,),(9,),(10,),(11,), (12,),(13,),(14,),(15,),(16,),(17,),(18,),(19,),(20,),(21,)
        ]
    }
]

mlp = GridSearchCV(MLPClassifier(max_iter=1000), param_grid, cv=3, scoring='accuracy')
mlp.fit(X_train, y_train)

print("Best parameters set found on development set:")
print(mlp.best_params_)
'''
