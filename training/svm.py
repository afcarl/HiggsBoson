from numpy import *
import scipy as sp
from sklearn import svm
from sklearn import cross_validation as cv
from sklearn import linear_model as lm
import pybrain as pb
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy import stats
import pickle
from sklearn.metrics.metrics import average_precision_score
LABEL = 32;


def loadTrainingData():
    return loadtxt('../Data/training_numeric.csv', delimiter=',')

def loadTestData():
    return loadtxt('../Data/test.csv_numeric', delimiter=',')


# Try SVM first
def classifySVM(trainData, percent = 1):
    dataRange = range(int( shape(trainData)[0]*percent))
    clf = svm.SVC();
    # use only the derived data
    clf.fit(trainData[dataRange,1:(13+1)], trainData[dataRange,32])
    print "Done Training the Data"
    y_predict = clf.predict(trainData[dataRange,1:(13+1)]);
    averagePrecision = average_precision_score(trainData[dataRange,32], y_predict);
    print "Scores "
    print averagePrecision;
    return [clf,y_predict];
    
def svmCV(trainData, fold=2):
    clf = svm.SVC();
    scores = cv.cross_val_score(clf, trainData[:,1:14], trainData[:,LABEL])
    print scores

if __name__ == '__main__':
    trainData = loadTrainingData();
    print "Data Loaded"
    numRows = shape(trainData)[0]
    subsetData = trainData[0:int(numRows*0.1)]
    #[svmClassifier, predictions] = classifySVM(trainData, 0.1)
    svmCV(subsetData)