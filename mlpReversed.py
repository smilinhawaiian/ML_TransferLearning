import time
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def get_data(csv):
    """
    Helper method to extract data from CSV files. Note that first row containing headers was deleted to make this work

    Parameters
        csv: String that should be 'mnist_train.csv' for training data or 'mnist_test.csv' for testing data

    Returns
        Matrix of data
    """
    # Capped at 70000 to equate with training/test for digits, plus my system can't load full letters CSV
    data = np.loadtxt(csv, delimiter=",", max_rows=70000)
    return data


def time_training(mlp, x, y):
    start = time.time()
    result = mlp.fit(x, y)
    end = time.time()
    return (result, end - start)


def get_fitness(mlp, x, y):
    results = mlp.fit(x, y)


# Get Data
digitsTrainData = get_data("data-csv/mnist-train-set.csv")
digitsTrainY = digitsTrainData[:, 0]
digitsTrainX = digitsTrainData[:, 1:]
print("got digits training data")
digitsTestData = get_data("data-csv/mnist-test-set.csv")
digitsTestY = digitsTestData[:, 0]
digitsTestX = digitsTestData[:, 1:]
print("got digits test data")
lettersData = get_data(
    "data-csv/AZ-handwritten-set.csv/AZ-handwritten-set.csv")
lettersY = lettersData[:, 0]
lettersX = lettersData[:, 1:]
X_train, X_test, y_train, y_test = train_test_split(
    lettersX, lettersY, stratify=lettersY, random_state=1, train_size=60000)  # Keep training sizes equal
print("got digits test and training data")


numTrails = 10
# Compare digits w/ and w/o letters training prior
transferTimes = []
baselineTimes = []
for i in range(numTrails):
    digits = MLPClassifier((196,)).fit(digitsTrainX, digitsTrainY)
    lettersTransfered = MLPClassifier((196, 10))
    lettersTransfered.coefs_ = digits.coefs_
    lettersOnly = MLPClassifier((196, 10))
    lettersTransfered, trainingTime = time_training(
        lettersTransfered, X_train, y_train)
    transferTimes.append(trainingTime)
    print("Training time for letters w/digits transfer: {}s".format(trainingTime))
    lettersOnly, trainingTime2 = time_training(
        lettersOnly, X_train, y_train)
    print("Training time for letters w/o transfer: {}s".format(trainingTime2))
    baselineTimes.append(trainingTime2)

print(transferTimes)
print(baselineTimes)
