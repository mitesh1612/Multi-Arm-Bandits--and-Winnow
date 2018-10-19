import csv
import numpy as np 
import sys
import os
import matplotlib.pyplot as plt
from scipy import stats

N = 100
NUM_TRIALS = 10000

missClassificationsPerceptron = np.array([[0.0]  * NUM_TRIALS] * N)
missClassificationsWinnow = np.array([[0.0]  * NUM_TRIALS] * N)

def readEmailSpamData(fileName):
	f = open(fileName)
	dataRaw = list(csv.reader(f))
	y = []
	temp = map(float,dataRaw[0][:-1])
	x = np.array(temp,dtype=float)
	y.append(int(dataRaw[0][-1]))
	for i in xrange(len(dataRaw)):
		temp = map(float,dataRaw[i][:-1])
		x = np.vstack((x,np.array(temp,dtype=float)))
		y.append(int(dataRaw[i][-1]))
	y = np.array(y)
	y[y==0] += -1	# Convert 0 to -1 for Perceptron and Winnow
	return x,y

def perceptronAlgorithm(x,y,eta,indices,n):
	global missClassificationsPerceptron
	w = np.zeros(len(x[0]))
	b = 0.0
	indices = np.random.uniform(size=NUM_TRIALS) * len(x)
	indices = map(int,indices)
	cnt = 0
	missClassifications = 0
	for i in indices:
		if (np.dot(w,x[i])+b)*y[i] <= 0:
			w = w + eta*y[i]*x[i]
			b = b + eta*y[i]
			missClassifications += 1
		missClassificationsPerceptron[n][cnt] += missClassifications
		cnt += 1
	return w,b

def winnowAlgorithm(x,y,eta,indices,n):
	global missClassificationsWinnow
	w = np.zeros(len(x[0]))
	w += float(1)/float(len(x[0]))
	cnt = 0
	missClassifications = 0
	for i in indices:
		if (np.dot(w,x[i]))*y[i] <= 0:
			Z = np.sum((w*np.exp(eta*y[i]*x[i])))
			w = w * np.exp(eta*y[i]*x[i])
			w = w / Z
			missClassifications += 1
		missClassificationsWinnow[n][cnt] += missClassifications
		cnt += 1
	return w

def cleanLists():
	global missClassificationsPerceptron
	global missClassificationsWinnow
	missClassificationsPerceptron = np.array([[0.0]  * NUM_TRIALS] * N)
	missClassificationsWinnow = np.array([[0.0]  * NUM_TRIALS] * N)

def averageAndPlotLists(etaVal):
	global missClassificationsPerceptron
	global missClassificationsWinnow
	meanPercept = np.mean(missClassificationsPerceptron,axis=0)
	meanWinnow = np.mean(missClassificationsWinnow,axis=0)
	stdErrPercept = stats.sem(missClassificationsPerceptron,axis=0)
	stdErrWinnow = stats.sem(missClassificationsWinnow,axis=0)
	plt.plot(range(1,NUM_TRIALS+1),meanPercept,c="red",label="Perceptron")
	plt.plot(range(1,NUM_TRIALS+1),meanWinnow,c="green",label="Winnow")
	plt.ylabel("Number of Misclassifications")
	plt.xlabel("Number of Trials")
	plt.legend(loc="best")
	plt.suptitle("Comparision of Winnow v/s Perceptron for ETA = "+str(etaVal))
	plt.show()
	plt.errorbar(range(1,NUM_TRIALS+1),meanPercept,yerr=stdErrPercept)
	plt.ylabel("Number of Misclassifications")
	plt.xlabel("Number of Trials")
	plt.suptitle("Perceptron for ETA = "+str(etaVal)+" (with Error Bars)")
	plt.show()
	plt.errorbar(range(1,NUM_TRIALS+1),meanWinnow,yerr=stdErrWinnow)
	plt.ylabel("Number of Misclassifications")
	plt.xlabel("Number of Trials")
	plt.suptitle("Winnow for ETA = "+str(etaVal)+" (with Error Bars)")
	plt.show()


if len(sys.argv) < 2:
	print "Enter the path for the Spam Base Data File"
	print "Run the File as Question2Part1.py <SpamBaseDataFilePath>"
	sys.exit(1)
fileName = sys.argv[1]
trainingData, trainingLabels = readEmailSpamData(fileName)
trainingData = np.array(trainingData)
trainingLabels = np.array(trainingLabels)
winnowTrainingData = list()
for i in xrange(len(trainingData)):
	x = trainingData[i]
	x1 = trainingData[i] * -1.0
	x1 = np.append(x1,[1.0])
	x1 = np.append(x1,[-1.0])
	winnowTrainingData.append(np.append(x,x1))
winnowTrainingData = np.array(winnowTrainingData)
etaValues = [0.1]
for etaVal in etaValues:
	print "Value of Eta: ",etaVal
	for n in xrange(N):		
		if n % 10 == 0:
			print "\t\tExperiment Done",n,"times."
		# Generating Indices
		indices = np.random.uniform(size=NUM_TRIALS) * len(trainingData)
		indices = map(int,indices)
		# Training the Perceptron
		w, b = perceptronAlgorithm(trainingData,trainingLabels,etaVal,indices,n)
		# Training Winnow's Algorithm
		w = winnowAlgorithm(winnowTrainingData,trainingLabels,etaVal,indices,n)
	averageAndPlotLists(etaVal)
	cleanLists()