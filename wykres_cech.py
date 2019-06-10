from sklearn import datasets
import matplotlib.pyplot as plt


iris = datasets.load_iris()
X = iris['data']
y = iris['target']

# - - - - - WYKRESY - - - - - #
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

fig1.suptitle('Dependences of selected features')
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

plt.show()
# - - - - - - - - - - - - - - - #