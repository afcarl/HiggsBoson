import random,string,math,csv,pandas
import numpy as np
import matplotlib.pyplot as plt
TRAIN_RATIO = 0.9;


raw = list(csv.reader(open("../Data/training.csv","rb"), delimiter=','))    
header = np.array(raw[0][1:-2])

xs = np.array([map(float, row[1:-2]) for row in raw[1:]])
(numPoints,numFeatures) = xs.shape

sSelector = np.array([row[-1] == 's' for row in raw[1:]])
bSelector = np.array([row[-1] == 'b' for row in raw[1:]])

weights = np.array([float(row[-2]) for row in raw[1:]])
labels = np.array([row[-1] for row in raw[1:]])
sumWeights = np.sum(weights)
sumSWeights = np.sum(weights[sSelector])
sumBWeights = np.sum(weights[bSelector])

#randomPermutation = random.sample(range(len(xs)), len(xs))
#np.savetxt("randomPermutation.csv",randomPermutation,fmt='%d',delimiter=',')
randomPermutation = np.array(map(int,np.array(list(csv.reader(open("../Util/randomPermutation.csv","rb"), delimiter=','))).flatten()))

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



###################################################

def AMS(s,b):
    assert s >= 0
    assert b >= 0
    bReg = 10.
    return math.sqrt(2 * ((s + b + bReg) * 
                          math.log(1 + s / (b + bReg)) - s))
    

validationScoresText = list(csv.reader(open("../MultiBoost-Build/scoresValidation.txt","rb"), delimiter=','))
validationScores = np.array([float(score[0]) for score in validationScoresText])

tIIs = validationScores.argsort()

wFactor = 1.* numPoints / numPointsValidation

s = np.sum(weightsValidation[sSelectorValidation])
b = np.sum(weightsValidation[bSelectorValidation])
amss = np.empty([len(tIIs)])
amsMax = 0
threshold = 0.0
for tI in range(len(tIIs)):
    # don't forget to renormalize the weights to the same sum 
    # as in the complete training set
    amss[tI] = AMS(max(0,s * wFactor),max(0,b * wFactor))
    # careful with small regions, they fluctuate a lot
    if tI < 0.9 * len(tIIs) and amss[tI] > amsMax:
        amsMax = amss[tI]
        threshold = validationScores[tIIs[tI]]
        #print tI,threshold
    if sSelectorValidation[tIIs[tI]]:
        s -= weightsValidation[tIIs[tI]]
    else:
        b -= weightsValidation[tIIs[tI]]

print "Final Value of Threshold"
print threshold
print ""

## AMS VS the rank        
fig = plt.figure()
fig.suptitle('MultiBoost AMS curves - AMS vs. Rank', fontsize=14, fontweight='bold')
vsRank = fig.add_subplot(111)
fig.subplots_adjust(top=0.85)

vsRank.set_xlabel('rank')
vsRank.set_ylabel('AMS')

vsRank.plot(amss,'b-')

vsRank.axis([0,len(amss), 0, 4])

plt.show()

## AMS VS the score
fig = plt.figure()
fig.suptitle('MultiBoost AMS curves - AMS vs. Score', fontsize=14, fontweight='bold')
vsScore = fig.add_subplot(111)
fig.subplots_adjust(top=0.85)

vsScore.set_xlabel('score')
vsScore.set_ylabel('AMS')

vsScore.plot(validationScores[tIIs],amss,'b-')

vsScore.axis([validationScores[tIIs[0]],validationScores[tIIs[-1]] , 0, 4])

plt.show()
