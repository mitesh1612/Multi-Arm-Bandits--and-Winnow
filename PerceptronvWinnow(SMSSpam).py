import numpy as np
import os,sys
import matplotlib.pyplot as plt
import re
from scipy import stats

if len(sys.argv) < 3:
	print "Add the Data File and StopWords File as Command Line Parameters"
	print "Run As python Question2Part2.py <DataFile> <stopWords>"
	sys.exit(1)

# SPAM = 1 HAM = -1
stopWords = set()
f = open(sys.argv[2])
for line in f:
	stopWords.add(line.strip())
f.close()

NUM_TRIALS = 10000
N = 100

messages = []
vocabulary = []
trainingLabels = []
regExp = re.compile(r"\W")

missClassificationsPerceptron = np.array([[0.0]  * NUM_TRIALS] * N)
missClassificationsWinnow = np.array([[0.0]  * NUM_TRIALS] * N)

f = open(sys.argv[1])
for line in f:
	line = line.split("\t")
	if line[0].startswith("ham"):
		trainingLabels.append(-1)
	elif line[0].startswith("spam"):
		trainingLabels.append(1)
	messages.append(line[1])

trainingLabels = np.array(trainingLabels)

def cleanText(line):
	line = regExp.sub(" ",line)
	line = line.lower()
	words = line.split()
	words = [word for word in words if word not in stopWords]
	return words


def buildVocabulary():
	global vocabulary
	for message in messages:
		words = cleanText(message)
		vocabulary.extend(words)
	vocabulary = sorted(list(set(vocabulary)))

def buildBagOfWords():
	bow = np.zeros([len(messages),len(vocabulary)])
	for i in xrange(len(messages)):
		words = cleanText(messages[i])
		for sentenceWord in words:
			for j, vocabWord in enumerate(vocabulary):
				if sentenceWord == vocabWord:
					bow[i,j] += 1
	return np.array(bow)

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

buildVocabulary()
trainingData = buildBagOfWords()
print "Bag of Words Created."
etaValues = [0.1]
winnowTrainingData = list()
for i in xrange(len(trainingData)):
	x = trainingData[i]
	x1 = trainingData[i] * -1.0
	x1 = np.append(x1,[1.0])
	x1 = np.append(x1,[-1.0])
	winnowTrainingData.append(np.append(x,x1))
winnowTrainingData = np.array(winnowTrainingData)

for etaVal in etaValues:
	print "Value of ETA: ",etaVal
	for n in xrange(N):		
		# Generating Indices
		indices = np.random.uniform(size=NUM_TRIALS) * len(trainingData)
		indices = map(int,indices)
		# Training the Perceptron
		w, b = perceptronAlgorithm(trainingData,trainingLabels,etaVal,indices,n)
		# Training Winnow's Algorithm
		w = winnowAlgorithm(winnowTrainingData,trainingLabels,etaVal,indices,n)
		if n % 10 == 0:
			print "\t\tExperiment Done: ",n,"Times."
	averageAndPlotLists(etaVal)
	cleanLists()