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
    #Capped at 70000 to equate with training/test for digits, plus my system can't load full letters CSV
    data = np.loadtxt(csv,delimiter=",", max_rows=70000) 
    return data

def time_training(mlp, x, y):
    start = time.time()
    result = mlp.fit(x, y)
    end = time.time()
    return (result, end - start)

def get_fitness(mlp, x, y):
    results = mlp.fit(x,y)
    

#Get Data
digitsTrainData = get_data("data/mnist-in-csv/mnist_train.csv") 
digitsTrainY = digitsTrainData[:,0]
digitsTrainX= digitsTrainData[:,1:]
print("got digits training data")
digitsTestData = get_data("data/mnist-in-csv/mnist_test.csv") 
digitsTestY = digitsTestData[:,0]
digitsTestX= digitsTestData[:,1:]
print("got digits test data")
lettersData = get_data("data/A_Z Handwritten Data.csv")
lettersY = lettersData[:,0]
lettersX= lettersData[:,1:]
X_train, X_test, y_train, y_test = train_test_split(lettersX, lettersY, stratify=lettersY,random_state=1, train_size=60000) #Keep training sizes equal
print("got digits test and training data")


numTrails = 10
#Compare digits w/ and w/o letters training prior
transferTimes = []
baselineTimes = []
for i in range(numTrails):
    letters = MLPClassifier((196,)).fit(X_train, y_train)
    digitsTransfered = MLPClassifier((196,26))
    digitsTransfered.coefs_ = letters.coefs_ #Use weights from letters MLP as the "transfer" of information from 
    digitsOnly = MLPClassifier((196,26))
    digitsTransfered, trainingTime = time_training(digitsTransfered, digitsTrainX, digitsTrainY)
    transferTimes.append(trainingTime)
    print("Training time for digits w/letters transfer: {}s".format(trainingTime))
    digitsOnly, trainingTime2 = time_training(digitsOnly, digitsTrainX, digitsTrainY)
    print("Training time for digits w/o transfer: {}s".format(trainingTime2))
    baselineTimes.append(trainingTime2)
print(transferTimes)
print(baselineTimes)

