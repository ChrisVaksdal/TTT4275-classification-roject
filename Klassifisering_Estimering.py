import numpy as np

def ones_divide(filename, test_set, featur):  # add ones to the data, and divide into test and trial data
    iris = np.loadtxt(filename, dtype='float_', delimiter=',')
    iris = np.vstack(iris)  # vstak, so it is easier to add ones

    ones = np.ones(len(iris))  # matrix of ones
    ones = ones.reshape(-1,1)  # reshape so it macth Iris

    add_ones = np.append(iris, ones, axis=1)  # addes ones to iris

    iris_test = add_ones[:test_set]
    iris_training = add_ones[test_set:]

    return iris_test, iris_training


D = 4+1  #features
C = 3  #classes
no_test = 20  # number of tests
setosa_test, setosa_training = ones_divide(".\\Iris_TTT4275\\class_1", no_test, D)
versicolor_test, versicolor_training = ones_divide(".\\Iris_TTT4275\\class_2", no_test, D)
virginica_test, virginica_training = ones_divide(".\\Iris_TTT4275\\class_3", no_test, D)

iris_label = np.loadtxt(".\\Iris_TTT4275\\iris.data", dtype='S', delimiter=',')  # 50 sampels pr flower
label = iris_label[:,0:3].astype('float_')