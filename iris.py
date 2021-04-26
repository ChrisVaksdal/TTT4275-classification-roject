import numpy as np
from sklearn.linear_model import SGDClassifier as Classifier
from sklearn.preprocessing import StandardScaler as Scaler
from sklearn.pipeline import make_pipeline




def importFile(path):
    return np.genfromtxt(path, dtype="float", delimiter=",")

def addColumn(arr, col):
    return np.append(arr, col, axis=1)

def addOnes(arr):
    col = [[1]] * len(arr)
    return addColumn(arr, col)

def splitTrainingTesting(data, numTraining):
    training = data[:numTraining]
    testing = data[numTraining:]
    return training, testing

def generateDataSet(path):
    data = importFile(path)
    data = addOnes(data)
    training, testing = splitTrainingTesting(data, int((len(data)/2)*3))
    return training, testing






if __name__ == "__main__":

    CLASS1_FILE = "iris/class_1"
    CLASS2_FILE = "iris/class_2"
    CLASS3_FILE = "iris/class_3"

    class1Training, class1Testing = generateDataSet(CLASS1_FILE)
    class2Training, class2Testing = generateDataSet(CLASS2_FILE)
    class3Training, class3Testing = generateDataSet(CLASS3_FILE)

    clf = make_pipeline(Scaler(), Classifier(max_iter=1000, tol=1e3))
    clf.fit(class1Training, class1Testing)

