import random,string,math,csv,pandas
import numpy as np
import matplotlib.pyplot as plt

TRAIN_RATIO = 0.9;

def DataToArff(xs,labels,weights,header,title,fileName):
    outFile = open(fileName + ".arff","w")
    outFile.write("@RELATION " + title + "\n\n")
    #header
    for feature in header:
        outFile.write("@ATTRIBUTE " + feature + 
                      " NUMERIC\n")
    outFile.write("@ATTRIBUTE classSignal NUMERIC\n")
    outFile.write("@ATTRIBUTE classBackground NUMERIC\n")
    outFile.write("\n@DATA\n")
    for x,label,weight in zip(xs,labels,weights):
        for xj in x:
            outFile.write(str(xj) + ",")
        if label == 's':
            outFile.write(str(weight) + "," + str(-weight) + "\n")
        else:
            outFile.write(str(-weight) + "," + str(weight) + "\n")
    outFile.close()

all = list(csv.reader(open("../Data/training.csv","rb"), delimiter=','))    
header = np.array(all[0][1:-2])

xs = np.array([map(float, row[1:-2]) for row in all[1:]])
(numPoints,numFeatures) = xs.shape

sSelector = np.array([row[-1] == 's' for row in all[1:]])
bSelector = np.array([row[-1] == 'b' for row in all[1:]])

weights = np.array([float(row[-2]) for row in all[1:]])
labels = np.array([row[-1] for row in all[1:]])
sumWeights = np.sum(weights)
sumSWeights = np.sum(weights[sSelector])
sumBWeights = np.sum(weights[bSelector])

randomPermutation = random.sample(range(len(xs)), len(xs))
np.savetxt("randomPermutation.csv",randomPermutation,fmt='%d',delimiter=',')
#randomPermutation = np.array(map(int,np.array(list(csv.reader(open("randomPermutation.csv","rb"), delimiter=','))).flatten()))

numPointsTrain = int(numPoints*TRAIN_RATIO);
numPointsValidation = numPoints - numPointsTrain

xsTrain = xs[randomPermutation[:numPointsTrain]]
xsValidation = xs[randomPermutation[numPointsTrain:]]

sSelectorTrain = sSelector[randomPermutation[:numPointsTrain]]
bSelectorTrain = bSelector[randomPermutation[:numPointsTrain]]
sSelectorValidation = sSelector[randomPermutation[numPointsTrain:]]
bSelectorValidation = bSelector[randomPermutation[numPointsTrain:]]

weightsTrain = weights[randomPermutation[:numPointsTrain]]
weightsValidation = weights[randomPermutation[numPointsTrain:]]

labelsTrain = labels[randomPermutation[:numPointsTrain]]
labelsValidation = labels[randomPermutation[numPointsTrain:]]

sumWeightsTrain = np.sum(weightsTrain)
sumSWeightsTrain = np.sum(weightsTrain[sSelectorTrain])
sumBWeightsTrain = np.sum(weightsTrain[bSelectorTrain])


DataToArff(xsTrain,labelsTrain,weightsTrain,header,"HiggsML_challenge_train","training")
DataToArff(xsValidation,labelsValidation,weightsValidation,header,"HiggsML_challenge_validation","validation")


testText = list(csv.reader(open("../Data/test.csv","rb"), delimiter=','))
testIds = np.array([int(row[0]) for row in testText[1:]])
xsTest = np.array([map(float, row[1:]) for row in testText[1:]])
weightsTest = np.repeat(1.0,len(testText)-1)
labelsTest = np.repeat('s',len(testText)-1)
DataToArff(xsTest,labelsTest,weightsTest,header,"HiggsML_challenge_test","test")