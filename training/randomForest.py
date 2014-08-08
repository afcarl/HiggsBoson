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

if __name__ == '__main__':
    pass;