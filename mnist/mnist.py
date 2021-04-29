"""
    File: mnist.py
    Author: Amanda Thunes Truyen & Christoffer-Robin Vaksdal
    Date: 2021-04-27

    Description: 

"""
import matplotlib.pyplot as plt
import numpy as np

from sklearn.neighbors import KNeighborsClassifier as Classifier
from sklearn.metrics import accuracy_score, plot_confusion_matrix
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split



def saveDigitsToFile(numDigits, digits, targets, identifier):
    _, axes = plt.subplots(nrows=1, ncols=numDigits, figsize=(10, 3))
    for ax, image, label in zip(axes, digits, targets):
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title("{}: {}".format(identifier, label))
    plt.savefig("{}.png".format(identifier))
    plt.close()

def main():

    mnist_data = load_digits()

    images = mnist_data.images
    targets = mnist_data.target

    NUMCHUNKS = 10

    nSamples = len(images)
    
    data = images.reshape((nSamples, -1))   # Flatten images from 2D (8, 8)into 1D arrays.

    XTrain, XTest, YTrain, YTest = train_test_split(data, targets, test_size=0.5, shuffle=False)

    clf = Classifier(n_neighbors=5 , metric="minkowski", p=2, n_jobs=1)
    clf.fit(XTrain, YTrain)

    plot_confusion_matrix(clf, XTest, YTest, values_format='g')
    plt.savefig("confusion.png")
    plt.close()

    predictions = clf.predict(XTest)
    score = accuracy_score(YTest, predictions)
    print("Score: {}".format(score))

    n = 4
    m = 4
    t = n + len(XTrain)
    f = n-m + len(XTrain)

    saveDigitsToFile(m, images[f : t], predictions[n-n:n], "prediction")   # Plot some digits.
    saveDigitsToFile(m, images[f : t], YTest[n-m:n], "example")   # Plot some digits.




if __name__ == "__main__":
    main()