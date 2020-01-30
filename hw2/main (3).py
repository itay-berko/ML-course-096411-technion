import datetime
from collections import OrderedDict

from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler

from utils import load_mnist
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.datasets.samples_generator import make_blobs
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


def func(x):
    return 3 + 3 * x + 3 * x ** 2


def grad(x):
    return 3 + 6 * x


def updt(grad, x, a):
    return x - a * grad(x)


def ex3a(x_train, y_train):
    # normalize values
    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train)

    model = SVC(kernel='linear')
    model.fit(x_train_scaled, y_train.values.ravel())
    return model


def ex3b(model, x_train, y_train, x_test, y_test):
    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    y_pred = model.predict(x_train_scaled)
    print(f'{datetime.datetime.now()} - train confusion matrix')
    print(confusion_matrix(y_train, y_pred))
    ls = model.score(x_train_scaled, y_train)
    print(f'{datetime.datetime.now()} - train error = {ls}')

    print(f'{datetime.datetime.now()} - test confusion matrix')
    y_pred = model.predict(x_test_scaled)
    print(confusion_matrix(y_test, y_pred))
    ld = model.score(x_test_scaled, y_test)

    print(f'{datetime.datetime.now()} - test error = {ld}')


def ex4(data):
    p_x = data["X"]
    p_y = data["y"]
    input_dim = len(p_x[0])
    w = np.zeros(input_dim)
    max_reps = 1000

    for t in range(max_reps):
        indicator = 1
        for i in range(len(p_x)):
            if np.dot(w, p_x[i]) * p_y[i] <= 0:
                w += p_x[i] * p_y[i]
                indicator = 0
                break
        if indicator == 1:
            return (w.transpose(), t)


def ex2a():
    x = range(-100, 100)
    y = [func(xi) for xi in x]
    plt.plot(x, y)
    plt.show()


def ex2b():
    print(grad)


def ex2c():
    x_min = -0.5
    return x_min


def test_perceptron():
    data = load_iris()
    p_x = np.array(data.data)
    p_y = np.array(data.target)
    # only the first class is separable
    selected_class = 0
    p_y[p_y != selected_class] = -1
    p_y[p_y == selected_class] = 1
    w,t = ex4({"X": p_x, "y": p_y})
    print(w,t)

if __name__ == '__main__':
    print(f'{datetime.datetime.now()} - load MNIST data')
    data_df, labels_df = load_mnist()
    X_train, X_test, y_train, y_test = train_test_split(data_df, labels_df, random_state=98, test_size=0.9)
    # X_train = np.array(X_train)
    # y_train = np.array(y_train.values.ravel())

    print(f'{datetime.datetime.now()} - ex2 create svm model')
    model = ex3a(X_train, y_train)

    print(f'{datetime.datetime.now()} - ex2 test model')
    ex3b(model, X_train, y_train, X_test, y_test)

    print(f'{datetime.datetime.now()} - ex2 test perceptron')
    test_perceptron()
    print('{datetime.datetime.now()} - ex2 finished')