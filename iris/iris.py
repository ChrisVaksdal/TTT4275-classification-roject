"""
    File: iris.py
    Author: Amanda Thunes Truyen & Christoffer-Robin Vaksdal
    Date: 2021-04-27

    Description: This file implements a series of SGD-classifiers, training and testing them on the SciKit Iris dataset.
                    The script will train and test several classifiers, plotting the confusion-matrices for each
                    It also plots histograms for each of the classifying features.

"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import SGDClassifier as Classifier
from sklearn.preprocessing import StandardScaler as Scaler
from sklearn.metrics import plot_confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_iris




"""
    Splits a given dataset into training and testing.
    Assumes dataset consists of three different classes in order.
    NUM_FIRST_PER_SET is how many samples in each set to be split into first part.
    NUM_TOTAL_PER_SET is how many samples in total per subset.
"""
def splitSet(data, NUM_FIRST_PER_SET, NUM_TOTAL_PER_SET=50):

    first = np.concatenate((
        data[                           : NUM_FIRST_PER_SET                         ], 
        data[ NUM_TOTAL_PER_SET         : NUM_TOTAL_PER_SET + NUM_FIRST_PER_SET     ], 
        data[ NUM_TOTAL_PER_SET * 2     : NUM_TOTAL_PER_SET * 2 + NUM_FIRST_PER_SET ]
    ))

    last = np.concatenate((
        data[ NUM_FIRST_PER_SET                         : NUM_TOTAL_PER_SET     ], 
        data[ NUM_TOTAL_PER_SET + NUM_FIRST_PER_SET     : NUM_TOTAL_PER_SET * 2 ],
        data[ NUM_TOTAL_PER_SET * 2 + NUM_FIRST_PER_SET : NUM_TOTAL_PER_SET * 3 ]
    ))

    return first, last


"""
    Removes a subset (column) at given index from given data.
"""
def removeSubset(data, index):
    indices = np.arange(len(data[0]))
    indices = np.delete(indices, index)
    return data[:,indices]


"""
    Plots confusion-matrix with given classifier, X and Y then saves it to png with given filename.
"""
def saveConfusionMatrixToFile(clf, X, Y, fileName, cmap=plt.cm.Oranges):
    view = plot_confusion_matrix(clf, X, Y, display_labels=[0,1,2], cmap=cmap, normalize=None)
    plt.savefig(fileName)
    plt.close()


"""
    Instantiates a classifier with given alpha and maximum number of iterations.
    Trains and tests classifier with given data.
    Plots and saves confusion matrices.
    Prints accuracy/score of classifier.
"""
def confusion(alpha, max_iter, XTrain, XTest, YTrain, YTest, fileNameSlug):
    clf = Classifier(alpha=alpha, max_iter=max_iter)  # Instantiate a Stochastic-Gradient-Descent-Classifer.
    clf.fit(XTrain, YTrain) # Fit classifier to data (let classifier learn from model).

    # Plot and save confusion-matrices:
    saveConfusionMatrixToFile(clf, XTrain, YTrain, "iris_confusion_matrix_training_{}.png".format(fileNameSlug))
    saveConfusionMatrixToFile(clf, XTest, YTest, "iris_confusion_matrix_testing_{}.png".format(fileNameSlug))

    # Print accuracy/score of classifier:
    print("Training data accuracy: {}".format(clf.score(XTrain, YTrain)))
    print("Testing data accuracy: {}".format(clf.score(XTest, YTest)))


"""
    Plots histograms for a given number of features and classes based on given data.
    Minimum, maximum and stepSize decides the number and widhts of the bins.
    figSizeCM is the total size of the figure in cm.
"""
def plotHistograms(data, minimum=0, maximum=8, stepSize=70, figSizeCM=15, numFeatures=4, numClasses=3, color="orange"):

    bins = np.arange(minimum, maximum, maximum/stepSize)    # Define x-axis of histogram.

    # Iterate over all features, plotting numClasses histograms for each:
    for feat in range(numFeatures):
        fig, axs = plt.subplots(numClasses, 1, figsize=(figSizeCM*(1/2.54), figSizeCM*(1/2.54)))

        # Iterate over all classes, plotting one histogram for each:
        for i, ax in enumerate(axs.flat):
            ax.hist(data.data[data.target == i, feat], bins=bins, color=color)
            ax.set_xlim(minimum, maximum)
            ax.set_ylabel("Class {}".format(i))

        fName = data.feature_names[feat]
        plt.xlabel(fName)
        plt.savefig("histogram_feature_{}.png".format(fName))
        plt.close()


"""
    Trains and tests a classifier with given values many times.
    Returns standard deviation of training- and testing-scores.
"""
def clfStdDev(XTrain, XTest, YTrain, YTest, alpha, iterations=20):

    trainingVals = np.zeros(iterations)
    testingVals = np.zeros(iterations)

    # Train and score classifier iterations number of times:
    for i in range(iterations):
        clf = Classifier(alpha=alpha, max_iter=100)  # Instantiate a Stochastic-Gradient-Descent-Classifer.
        clf.fit(XTrain, YTrain) # Fit classifier to data (let classifier learn from model).
        trainingVals[i] = clf.score(XTrain, YTrain)
        testingVals[i] = clf.score(XTest, YTest)
    
    # Calculate standard deviations:
    trainingDev = np.std(trainingVals)
    testingDev = np.std(testingVals)

    return trainingDev, testingDev




def main():

    MAX_ITER = 100  # Maximum number of iterations used by classifier.

    # Get iris-dataset from sklearn:
    irisData = load_iris()
    X = irisData.data
    Y = irisData.target


    # Get confusion matrix and accuracy using first 30 in dataset as training and try different values for alpha:
    XTrain, XTest = splitSet(X, 30)
    YTrain, YTest = splitSet(Y, 30)

    print("Trying different values for alpha")
    alphas = [0.002 + 0.002*i for i in range(int(0.03/0.002))]
    for a in alphas:
        print("a = {}".format(a))
        confusion(a, MAX_ITER, XTrain, XTest, YTrain, YTest, "alpha_{}".format(a))

    ALPHA = 0.01
    # Get confusion matrix and accuracy using first 30 in dataset as training:
    XTrain, XTest = splitSet(X, 30)
    YTrain, YTest = splitSet(Y, 30)

    print("Using first 30 for learning.")
    confusion(ALPHA, MAX_ITER, XTrain, XTest, YTrain, YTest, "first_30")


    # ---||--- using last 30 in dataset as training:
    XTest, XTrain = splitSet(X, 20)
    YTest, YTrain = splitSet(Y, 20)

    print("Using last 30 for learning.")
    confusion(ALPHA, MAX_ITER, XTrain, XTest, YTrain, YTest, "last_30")


    # Plot histograms:
    plotHistograms(irisData)


    # Get confusion matrix and accuracy using the first 30 in dataset as training, but feature 1 (sepal-width) is removed:
    XTrain, XTest = splitSet(X, 30)
    YTrain, YTest = splitSet(Y, 30)
    
    XTest = removeSubset(XTest, 1)
    XTrain = removeSubset(XTrain, 1)

    print("Using first 30, but feature 1 is removed.")
    confusion(ALPHA, MAX_ITER, XTrain, XTest, YTrain, YTest, "feat_1_removed")


    # ---||---, but only features X and Y are used:
    XTrain, XTest = splitSet(X, 30)
    YTrain, YTest = splitSet(Y, 30)
    
    XTrain = removeSubset(XTrain, 0)
    XTest = removeSubset(XTest, 0)
    XTrain = removeSubset(XTrain, 1)
    XTest = removeSubset(XTest, 1)

    print("Using first 30, but only features 2 and 3 are used.")
    confusion(ALPHA, MAX_ITER, XTrain, XTest, YTrain, YTest, "2_features")


    # Calculate standard deviation when using first 30 for training and removing feature 1.
    STD_DEV_ITERATIONS = 100    # Number of times to retrain model before calculating std. dev.
    trainingDev, testingDev = clfStdDev(XTrain, XTest, YTrain, YTest, ALPHA, STD_DEV_ITERATIONS)
    print("Standard-deviation using first 30 and removing feature 1.\n{} Iterations".format(STD_DEV_ITERATIONS))
    print("Training: {}\nTesting: {}".format(trainingDev, testingDev))




if __name__ == "__main__":
    main()
