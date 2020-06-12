import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


def getCharData():

    # Import csv dataset
    dataset = pd.read_csv(
        './data-csv/AZ-handwritten-set.csv/AZ-handwritten-set.csv')

    # Shuffle dataset randomly (data is sorted by label)
    dataset = shuffle(dataset)
    dataset = dataset.values

    # Split dataset into inputs and labels
    images = dataset[:, 1:]
    labels = dataset[:, 0]

    # Split into train and test data after normalizing
    (trainImages, testImages, trainLabels, testLabels) = train_test_split(
        images/255.0, labels.astype("int"), test_size=0.25, random_state=42)

    # Encode labels
    ohe = OneHotEncoder()
    trainLabels = trainLabels.reshape(-1, 1)
    ohe.fit(trainLabels)
    trainLabels = ohe.transform(trainLabels)
    trainLabels = trainLabels.toarray()

    testLabels = testLabels.reshape(-1, 1)
    ohe.fit(testLabels)
    testLabels = ohe.transform(testLabels)
    testLabels = testLabels.toarray()

    return trainImages, trainLabels, testImages, testLabels


def getDigitsData():

    # Import csv dataset
    dataset = pd.read_csv(
        './data-csv/mnist-train-set.csv')

    # Shuffle dataset randomly (data is sorted by label)
    dataset = shuffle(dataset)
    dataset = dataset.values

    # Split dataset into inputs and labels
    trainImages = dataset[:, 1:]
    trainLabels = dataset[:, 0]

    # Import csv test dataset
    dataset = pd.read_csv(
        './data-csv/mnist-test-set.csv')

    # Shuffle dataset randomly (data is sorted by label)
    dataset = shuffle(dataset)
    dataset = dataset.values

    # Split dataset into inputs and labels
    testImages = dataset[:, 1:]
    testLabels = dataset[:, 0]

    # Encode labels
    ohe = OneHotEncoder()
    trainLabels = trainLabels.reshape(-1, 1)
    ohe.fit(trainLabels)
    trainLabels = ohe.transform(trainLabels)
    trainLabels = trainLabels.toarray()

    testLabels = testLabels.reshape(-1, 1)
    ohe.fit(testLabels)
    testLabels = ohe.transform(testLabels)
    testLabels = testLabels.toarray()

    return trainImages, trainLabels, testImages, testLabels


charTrainImages, charTrainLabels, charTestImages, charTestLabels = getCharData()
digitTrainImages, digitTrainLabels, digitTestImages, digitTestLabels = getDigitsData()

print("s")
