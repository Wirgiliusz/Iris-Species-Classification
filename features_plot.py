from sklearn import datasets
import matplotlib.pyplot as plt

# Load of data
iris = datasets.load_iris()
X = iris['data']    # wejscia
y = iris['target']  # wyjscia

# Lists for data
s_l = []    # sepal length
p_l = []    # petal length
s_w = []    # sepal width
p_w = []    # petal width
color = []  # colors
species_name = ['Setosa', 'Versicolor', 'Virginica', ''] # species

# Fill of data lists
for i in range(0, 150):
    s_l.append(X[i][0])
    p_l.append(X[i][2])
    s_w.append(X[i][1])
    p_w.append(X[i][3])
    # Assignment of colors
    if(y[i] == 0):
        color.append('ro')
    elif(y[i] == 1):
        color.append('go')
    elif(y[i] == 2):
        color.append('bo')

# Plotting 
fig1 = plt.figure(1)

sub1 = fig1.add_subplot(221)    
sub2 = fig1.add_subplot(222)
sub3 = fig1.add_subplot(223)
sub4 = fig1.add_subplot(224)

for i in range(0, 150):
    if(i==0):
        n = species_name[0]
    elif(i==50):
        n = species_name[1]
    elif(i==100):
        n = species_name[2]
    else:
        n = species_name[3]

    sub1.plot(s_l[i], p_l[i], color=color[i][0], marker=color[i][1], label=n)
    sub2.plot(s_w[i], p_w[i], color=color[i][0], marker=color[i][1], label=n)
    sub3.plot(s_l[i], s_w[i], color=color[i][0], marker=color[i][1], label=n)
    sub4.plot(p_w[i], p_l[i], color=color[i][0], marker=color[i][1], label=n)

# Plots descriptions
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
