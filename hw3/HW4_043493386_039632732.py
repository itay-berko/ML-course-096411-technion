import datetime

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_validate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Imputer
from sklearn.svm import SVC

from ex3.utils import load_mnist
import matplotlib.pyplot as plt


def cv(X, y, model, folds):
    kf = KFold(n_splits=folds, shuffle=False).split(X)
    folds_train_acc = []
    folds_test_acc = []

    for i, fold in enumerate(kf):
        print('{} - fold {}'.format(datetime.datetime.now(), i))
        curr_x_train = X[fold[0]]
        curr_y_train = y[fold[0]]
        curr_x_test = X[fold[1]]
        curr_y_test = y[fold[1]]
        model.fit(curr_x_train, curr_y_train)

        # get train accuracy
        y_train_pred = model.predict(curr_x_train)
        curr_train_acc = accuracy_score(curr_y_train, y_train_pred)
        folds_train_acc.append(curr_train_acc)

        # get test accuracy
        y_test_pred = model.predict(curr_x_test)
        curr_test_acc = accuracy_score(curr_y_test, y_test_pred)
        folds_test_acc.append(curr_test_acc)

    train_acc = np.mean(folds_train_acc)
    validation_acc = np.mean(folds_test_acc)
    print('{} - cv results for model: {},{}'.format(datetime.datetime.now(), train_acc, validation_acc))
    return [train_acc, validation_acc]


def get_test_error(model, x_train, y_train, x_test, y_test):
    print('{} - fitting model'.format(datetime.datetime.now()))
    model.fit(x_train, y_train)
    print('{} - testing model'.format(datetime.datetime.now()))
    y_pred = model.predict(x_test)
    test_error = accuracy_score(y_test, y_pred)
    print('{} - test error for model = {}'.format(datetime.datetime.now(), test_error))
    return test_error


def test_models(data, labels):
    x_train, x_test, y_train, y_test = train_test_split(data, labels, random_state=98, test_size=0.2)
    models_results = {}

    kernel_type = 'linear'
    print('{} - running kernel: {}'.format(datetime.datetime.now(), kernel_type))

    curr_model = SVC(kernel=kernel_type)
    acc = cv(x_train, y_train, curr_model, 5)
    models_results[kernel_type] = acc

    print('{} - running kernel: {}, fitting entire train set'.format(datetime.datetime.now(), kernel_type))
    test_error = get_test_error(curr_model, x_train, y_train, x_test, y_test)
    models_results[kernel_type].append(test_error)

    kernel_type = 'poly'
    poly_dict = {}
    print('{} - running kernel: {}'.format(datetime.datetime.now(), kernel_type))
    for deg in [2, 3, 4, 5, 6, 7, 8, 9]:
        print('{} - running kernel: {}, degree {}'.format(datetime.datetime.now(), kernel_type, deg))
        curr_model = SVC(kernel=kernel_type, degree=deg)
        acc = cv(x_train, y_train, curr_model, 5)
        poly_dict[deg] = acc[1]
        run_name = '{}_{}'.format(kernel_type, deg)
        models_results[run_name] = acc

        test_error = get_test_error(curr_model, x_train, y_train, x_test, y_test)
        models_results[run_name].append(test_error)
    plt.plot(poly_dict.keys(), poly_dict.values())
    plt.xlabel('polynom degrees')
    plt.ylabel('validation accuracy')
    plt.title('validation accuracy for polynomial kernels')
    plt.savefig("poly.png")
    # plt.show()

    # scale the data for rbf
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)

    plt.clf()
    kernel_type = 'rbf'
    rbf_acc = {}
    print('{} - running kernel: {}'.format(datetime.datetime.now(), kernel_type))
    for gamma in [0.001, 0.01, 0.1, 1, 10]:
        print('{} - running kernel: {}, gamma {}'.format(datetime.datetime.now(), kernel_type, gamma))
        curr_model = SVC(kernel=kernel_type, gamma=gamma)
        acc = cv(x_train, y_train, curr_model, 5)
        rbf_acc[gamma] = acc[1]
        run_name = '{}_{}'.format(kernel_type, gamma)
        models_results[run_name] = acc

        test_error = get_test_error(curr_model, x_train, y_train, x_test, y_test)
        models_results[run_name].append(test_error)
    plt.plot(rbf_acc.keys(), rbf_acc.values())
    plt.xlabel('rbf degrees')
    plt.ylabel('validation accuracy')
    plt.title('validation accuracy for rbf kernels')
    plt.savefig("rbf.png")
    # plt.show()

    for model_type, res in models_results.items():
        print('{}: {}'.format(model_type, res))

    cv_sort = sorted([(model_type, res[1]) for model_type, res in models_results.items()], key=lambda x: x[1])
    best_cv = cv_sort[-1]
    print('the {} model was best on cross validation with error {}'.format(best_cv[0], best_cv[1]))

    test_sort = sorted([(model_type, res[2]) for model_type, res in models_results.items()], key=lambda x: x[1])
    best_test = test_sort[-1]
    print('the {} model was best on test data with error {}'.format(best_test[0], best_test[1]))


if __name__ == '__main__':
    data_df, labels_df = load_mnist()
    data = np.array(data_df)
    labels = np.array(labels_df)
    test_models(data, labels)
