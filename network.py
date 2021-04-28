from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier  
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

import numpy as np
import matplotlib.pyplot as plt

# - - - - - CLASSIFIERS TESTS FOR DIFFERENT PARAMETERS - - - - - # 
# Loading iris data
iris = datasets.load_iris()
X = iris['data']
y = iris['target']

# Split of the data in two groups - training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Normalization and standarization of data
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


# - Perceptron - #
ppn = Perceptron(max_iter=40, eta0=0.1, random_state=0)
ppn.fit(X_train_std, y_train)
y_pred_ppn = ppn.predict(X_test_std)

# - Multi-layer perceptron - #
mlp = MLPClassifier(hidden_layer_sizes=(10, 10), activation="relu", solver='adam', max_iter=1000)
mlp.fit(X_train_std, y_train)  
y_pred_mlp = mlp.predict(X_test_std)

# - K closest neighbours - #
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_std, y_train)
y_pred_knn = knn.predict(X_test_std)

# Save results to file
file = open('Results.txt', 'w')

file.write("Comparison of real results with network predictions:\n")
file.write(str(y_test) + ' - real results\n')
file.write(str(y_pred_ppn) + ' - perceptron\n')
file.write(str(y_pred_mlp) + ' - multi-layer perceptron\n')
file.write(str(y_pred_knn) + ' - K closest neighbours\n\n')

file.write('Accuracy of perceptron: ' + str(round(accuracy_score(y_test, y_pred_ppn)*100,2)) + "%\n")
file.write('Accuracy of multi-layer perceptron: ' + str(round(accuracy_score(y_test, y_pred_mlp)*100,2)) + "%\n")
file.write('Accuracy of K closest neighbours: ' + str(round(accuracy_score(y_test, y_pred_knn)*100,2)) + "%\n\n")


# - Multi-layer perceptron - #
# Tests for different parameters #
NUMBER_OF_TESTS = 100
NUMBER_OF_ITERATIONS = 500

# Tests of various activation functions and learning algorithms
activation_solver_pairs = (("relu", "adam"), ("tanh", "adam"), ("relu", "sgd"), ("tanh", "sgd"))

for solver_name, activation_name in activation_solver_pairs:
    accuracy = 0
    for _ in range(0, NUMBER_OF_TESTS):
        mlp_classifier = MLPClassifier(hidden_layer_sizes=(10, 10), activation=solver_name, solver=activation_name, max_iter=NUMBER_OF_ITERATIONS)
        mlp_classifier.fit(X_train_std, y_train)  
        mlp_prediction = mlp_classifier.predict(X_test_std)
        accuracy += accuracy_score(y_test, mlp_prediction)
    
    average_accuracy = accuracy / NUMBER_OF_TESTS
    file.write("MLP [" + solver_name + ", " + activation_name + "] (10, 10): " + str(round(average_accuracy * 100, 2)) + "%\n")


# Tests of various numbers of hidden layers
hidden_layers_sizes = ((10,), (10, 10,), (10, 10, 10,), (10, 10, 10, 10,))

for layer_size in hidden_layers_sizes:
    accuracy = 0
    for _ in range(0, NUMBER_OF_TESTS):
        mlp_classifier = MLPClassifier(hidden_layer_sizes=layer_size, activation="relu", solver='adam', max_iter=NUMBER_OF_ITERATIONS)
        mlp_classifier.fit(X_train_std, y_train)  
        mlp_prediction = mlp_classifier.predict(X_test_std)
        accuracy += accuracy_score(y_test, mlp_prediction)
    
    average_accuracy = accuracy / NUMBER_OF_TESTS
    file.write("MLP [relu, adam] " + str(layer_size) + ": " + str(round(average_accuracy * 100, 2)) + "%\n")


# - Looking for best parameters - #
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
