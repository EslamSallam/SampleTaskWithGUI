import numpy as np
from matplotlib import pyplot as plt
import matplotlib.lines as mlines
from sklearn.metrics import confusion_matrix

ir_class = []
f1 = 0
f2 = 0

c1 = 0
c2 = 0
eta = 0
epochs = 0
add_bais = 0

data = np.empty([150, 5])
training = np.empty([60, 3])
test = np.empty([60, 3])
we = []




def setClassesAndFeatures(cc1, cc2, feature1, feature2):
    global c1
    global c2
    global f1
    global f2

    c1 = cc1
    c2 = cc2
    f1 = feature1
    f2 = feature2


def finalize_data(ee, epo, bais):
    global eta
    global epochs
    global add_bais
    eta = float(ee)
    epochs = int(epo)
    add_bais = int(bais)


def drawIris():
    plt.scatter(data[0:50, 0], data[0:50, 1], np.pi*5, (1, 0, 0))
    plt.scatter(data[50:100, 0], data[50:100, 1], np.pi*5, (0, 1, 0))
    plt.scatter(data[100:150, 0], data[100:150, 1], np.pi*5, (0, 0, 1))
    plt.show()

    plt.scatter(data[0:50, 0], data[0:50, 2], np.pi*5, (1, 0, 0))
    plt.scatter(data[50:100, 0], data[50:100, 2], np.pi*5, (0, 1, 0))
    plt.scatter(data[100:150, 0], data[100:150, 2], np.pi*5, (0, 0, 1))
    plt.show()

    plt.scatter(data[0:50, 0], data[0:50, 3], np.pi*5, (1, 0, 0))
    plt.scatter(data[50:100, 0], data[50:100, 3], np.pi*5, (0, 1, 0))
    plt.scatter(data[100:150, 0], data[100:150, 3], np.pi*5, (0, 0, 1))
    plt.show()

    plt.scatter(data[0:50, 1], data[0:50, 2], np.pi*5, (1, 0, 0))
    plt.scatter(data[50:100, 1], data[50:100, 2], np.pi*5, (0, 1, 0))
    plt.scatter(data[100:150, 1], data[100:150, 2], np.pi*5, (0, 0, 1))
    plt.show()

    plt.scatter(data[0:50, 1], data[0:50, 3], np.pi*5, (1, 0, 0))
    plt.scatter(data[50:100, 1], data[50:100, 3], np.pi*5, (0, 1, 0))
    plt.scatter(data[100:150, 1], data[100:150, 3], np.pi*5, (0, 0, 1))
    plt.show()

    plt.scatter(data[0:50, 2], data[0:50, 3], np.pi*5, (1, 0, 0))
    plt.scatter(data[50:100, 2], data[50:100, 3], np.pi*5, (0, 1, 0))
    plt.scatter(data[100:150, 2], data[100:150, 3], np.pi*5, (0, 0, 1))
    plt.show()


def read_data():
    global ir_class
    global data
    global training
    global test


    f = open("IrisData.txt", 'r')
    contents = f.read()
    con = contents.split("\n")
    for i, obj in enumerate(con):
        if i > 0:
            s = obj.split(",")
            if s[4] == 'Iris-setosa':
                data[i - 1, 4] = 0
            elif s[4] == 'Iris-versicolor':
                data[i - 1, 4] = 1
            elif s[4] == 'Iris-virginica':
                data[i - 1, 4] = 2
            data[i - 1, 0] = s[0]
            data[i - 1, 1] = s[1]
            data[i - 1, 2] = s[2]
            data[i - 1, 3] = s[3]
    drawIris()
    if add_bais == 1:
        data1 = np.ones([50, 4])
        data2 = np.ones([50, 4])

        data1[:, 1] = data[c1*50:(c1 + 1) * 50, f1]
        data1[:, 2] = data[c1*50:(c1 + 1) * 50, f2]
        data1[:, 3] = -1

        data2[:, 1] = data[c2*50:(c2 + 1) * 50, f1]
        data2[:, 2] = data[c2*50:(c2 + 1) * 50, f2]
        data2[:, 3] = 1

        training = np.ones([60, 4])
        test = np.ones([40, 4])
    else:
        data1 = np.ones([50, 3])
        data2 = np.ones([50, 3])

        data1[:, 0] = data[c1 * 50:(c1 + 1) * 50, f1]
        data1[:, 1] = data[c1 * 50:(c1 + 1) * 50, f2]
        data1[:, 2] = -1

        data2[:, 0] = data[c2 * 50:(c2 + 1) * 50, f1]
        data2[:, 1] = data[c2 * 50:(c2 + 1) * 50, f2]
        data2[:, 2] = 1

        training = np.ones([60, 3])
        test = np.ones([40, 3])

    training[:30, :], test[:20, :] = data1[:30, :], data1[30:50, :]
    training[30:60, :], test[20:40, :] = data2[:30, :], data2[30:50, :]

    np.random.shuffle(training)
    np.random.shuffle(test)

    x = training[:, 0:-1]
    y = training[:, -1]
    x_test = test[:, 0:-1]
    y_test = test[:, -1]
    train_data(x, y, x_test, y_test)



def train_data(X, Y, x_test, y_test):
    if add_bais == 1:
        W = np.random.rand(3)
    else:
        W = np.random.rand(2)
    er = 0

    for _ in range(epochs):
        for i, obj in enumerate(X):
            #np.float_(y)
            y = np.dot(W, obj)

            a = activation_fn(y)
            er = (Y[i] - a)
            W = W + eta * er * obj
    y_prediction = np.zeros(40)
    print(W)
    for i, obj in enumerate(x_test):
        y = W.T.dot(obj)
        a = activation_fn(y)
        y_prediction[i] = a
        print(a, "   right  ", y_test[i])

    accuracy = np.mean(y_prediction == y_test) * 100
    cm1 = confusion_matrix(y_test, y_prediction)
    print(accuracy)
    print(cm1)


    # pointy = [5, (-W[0]-(W[2]*5))/W[1]]
    # pointx = [-W[0]/W[2], 0]
    # plt.plot([pointx[0], pointy[0]], [pointx[1], pointy[1]], color='r')
    # plt.show()

    b = 0
    if add_bais == 1:
        plt.scatter(x_test[:, 1], x_test[:, 2], c=y_prediction)

        b = W[0]
        # point1 = [(-b -(7*W[2]))/ W[1], 7]
        # point2 = [0, -b / W[2]]
        point1 = [(-W[2] * x_test[20, 2] - b) / W[1], x_test[20, 2]]
        point2 = [x_test[20, 1], (-W[1] * x_test[20, 1] - b) / W[2]]
    else:
        plt.scatter(x_test[:, 0], x_test[:, 1], c=y_prediction)
        point1 = [ -W[1] * x_test[20, 1] / W[0], x_test[20, 1]]
        point2 = [ x_test[20, 0], -W[0] * x_test[20, 0] / W[1]]

    plt.plot([point1[0],point2[0]],[point1[1],point2[1]], color='r')
    plt.show()

    x2 = 0
    # plt.scatter(x_test[:, 0], y_test, 5*np.pi, (1, 0, 0))
    # newline((x_test[0, 1], y_test[0]), (x_test[10, 1], y_test[10]))
    # plt.show()

def activation_fn(x):
    # return (x >= 0).astype(np.float32)
    if x >= 0:
        return  1
    elif x < 0:
        return  -1



def newline(p1, p2):
    ax = plt.gca()
    xmin, xmax = ax.get_xbound()

    if(p2[0] == p1[0]):
        xmin = xmax = p1[0]
        ymin, ymax = ax.get_ybound()
    else:
        ymax = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmax-p1[0])
        ymin = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmin-p1[0])

    l = mlines.Line2D([xmin,xmax], [ymin,ymax])
    ax.add_line(l)
    return l
