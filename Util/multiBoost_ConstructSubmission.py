import csv
import numpy as np

threshold = 0.0
testText = list(csv.reader(open("../Data/test.csv","rb"), delimiter=','))
testIds = np.array([int(row[0]) for row in testText[1:]])
xsTest = np.array([map(float, row[1:]) for row in testText[1:]])
weightsTest = np.repeat(1.0,len(testText)-1)
labelsTest = np.repeat('s',len(testText)-1)

testScoresText = list(csv.reader(open("../MultiBoost-Build/scoresTest.txt", "rb"),delimiter=','))
testScores = np.array([float(score[0]) for score in testScoresText])

testInversePermutation = testScores.argsort()

testPermutation = list(testInversePermutation)
for tI,tII in zip(range(len(testInversePermutation)),
                  testInversePermutation):
    testPermutation[tII] = tI
    
submission = np.array([[str(testIds[tI]),str(testPermutation[tI]+1),
                       's' if testScores[tI] >= threshold else 'b'] 
            for tI in range(len(testIds))])

submission = np.append([['EventId','RankOrder','Class']],
                        submission, axis=0)

np.savetxt("submission.csv",submission,fmt='%s',delimiter=',')