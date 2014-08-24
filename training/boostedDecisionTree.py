import numpy as np
import random, string, math, csv
import matplotlib.pyplot as plt

ratio_train = 0.9;


raw = list(csv.reader(open("../Data/training.csv","rb"), delimiter=","))
X = np.array([map(float,row[1:-2]) for row in raw[1:]])

(numPoints, numFeatures) = X.shape;
sSelector = np.array([row[-1] == 's' for row in raw[1:]])
bSelector = np.array([row[-1] == 'b' for row in raw[1:]])
weights = np.array([float(row[-2]) for row in raw[1:]])
sumWeights = np.sum(weights)
sumSWeights = np.sum(weights[sSelector])
sumBWeights = np.sum(weights[bSelector])


# Splitting the data into training and cross-validation sets
randomPermutation = random.sample(range(len(X)), len(X))
numPointsTrain = int(numPoints*ratio_train)
numPointsCV = numPoints - numPointsTrain

Xtrain = X[randomPermutation[:numPointsTrain]]
XCV = X[randomPermutation[numPointsTrain:]]

sSelectorTrain = sSelector[randomPermutation[:numPointsTrain]]
sSelectorCV = sSelector[randomPermutation[numPointsTrain:]]

bSelectorTrain = bSelector[randomPermutation[:numPointsTrain]]
bSelectorCV = bSelector[randomPermutation[numPointsTrain:]]


weightsTrain = weights[randomPermutation[:numPointsTrain]]
weightsCV = weights[randomPermutation[numPointsTrain:]]

sumWeightsTrain = np.sum(weightsTrain)
sumSWeightsTrain = np.sum(weightsTrain[sSelectorTrain])
sumBWeightsTrain = np.sum(weightsTrain[bSelectorTrain])


## XtrainTranspose = Xtrain.transpose()

weightsBalancedTrain = np.array([(0.5* weightsTrain[i]/sumSWeightsTrain 
                                 if sSelectorTrain[i]
                                 else 0.5* weightsTrain[i]/sumBWeightsTrain 
                                 for i in range(numPointsTrain))])

