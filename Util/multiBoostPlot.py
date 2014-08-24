import random,string,math,csv,pandas
import numpy as np
import matplotlib.pyplot as plt



resultsText = list(csv.reader(open("../MultiBoost-Build/results.dta","rb"), delimiter='\t'))
ts = [int(result[0]) for result in resultsText]
trainErrors = np.array([float(result[5]) for result in resultsText])
testErrors = np.array([float(result[11]) for result in resultsText])

fig = plt.figure()
fig.suptitle('MultiBoost learning curves', fontsize=14, fontweight='bold')
ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.85)

ax.set_xlabel('number of boosting iterations')
ax.set_ylabel('balanced weighted error rate')

ax.annotate('training error', xy=(0.9*len(ts), trainErrors[len(ts)-1]), 
            xytext=(0.6*len(ts), trainErrors[len(ts)-1]-0.05),
            arrowprops=dict(facecolor='blue', shrink=0.05))
ax.annotate('validation error', xy=(0.9*len(ts), testErrors[len(ts)-1]), 
            xytext=(0.6*len(ts), testErrors[len(ts)-1]+0.05),
            arrowprops=dict(facecolor='red', shrink=0.05))

ax.plot(ts,trainErrors,'b-')
ax.plot(ts,testErrors,'r-')

ax.axis([0, len(ts), 0.1, 0.3])

plt.show()