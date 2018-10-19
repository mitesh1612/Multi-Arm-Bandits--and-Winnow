import numpy as np 
from scipy.stats import bernoulli
import matplotlib.pyplot as plt 
import random
from math import sqrt, log
from scipy import stats

N = 100
NUM_TRIALS = 10000
ALPHA = 4.0
ETA = 0.1

probArm1 = [0.9,0.9,0.55]
probArm2 = [0.6,0.8,0.45]
regretUCB = np.array([[0.0] * NUM_TRIALS] * N)
regretEXP3 = np.array([[0.0] * NUM_TRIALS] * N)
regretEGreedy = np.array([[0.0] * NUM_TRIALS] * N)
regretEGreedy2 = np.array([[0.0] * NUM_TRIALS] * N)
optimalArmsUCB = np.array([[0.0] * NUM_TRIALS] * N)
optimalArmsEXP3 = np.array([[0.0] * NUM_TRIALS] * N)
optimalArmsEGreedy = np.array([[0.0] * NUM_TRIALS] * N)
optimalArmsEGreedy2 = np.array([[0.0] * NUM_TRIALS] * N)

def updateMean(oldMean,currentReward,Ni):
	newMean = 0.0
	newMean = oldMean + float(currentReward-oldMean)/float(Ni)
	return newMean

def trainUCB(x1,x2,n):
	global regretUCB
	global optimalArmsUCB
	mean1 = 0.0
	mean2 = 0.0
	count1 = 1
	count2 = 1
	regret = 0
	for i in xrange(len(x1)):
		CB1 = sqrt((float(ALPHA) * log(i+1))/(2 * count1))
		CB2 = sqrt((float(ALPHA) * log(i+1))/(2 * count2))
		arm = np.argmax([mean1+CB1,mean2+CB2]) + 1
		if arm == 1:
			count1 += 1
			mean1 = updateMean(mean1,x1[i],count1)
			optimalArmsUCB[n][i] += 1
		else:
			count2 += 1
			mean2 = updateMean(mean2,x2[i],count1)
		regret += (x1[i]-eval('x'+str(arm)+'[i]'))
		regretUCB[n][i] += regret


def trainEXP3(x1,x2,n):
	global regretEXP3
	global optimalArmsEXP3
	probs = [0.5,0.5]
	l1,l2 = 0.0, 0.0
	l1Til, l2Til = 0.0, 0.0
	L1, L2 = 0.0, 0.0
	regret = 0
	for i in xrange(len(x1)):
		arm = np.random.choice(2,1,p=probs) + 1
		losses = [0.0] * 2
		if arm == 1:
			optimalArmsEXP3[n][i] += 1
			l1 = 1 - x1[i]
			l2 = 0
		else:
			l2 = 1 - x2[i]
			l1 = 0
		l1Til = l1/probs[0]
		l2Til = l2/probs[1]
		L1 += l1Til
		L2 += l2Til
		expSum = np.exp(-1.0 * ETA * L1) + np.exp(-1.0 * ETA * L2)
		probs[0] = np.exp(-1.0 * ETA * L1)/expSum
		probs[1] = np.exp(-1.0 * ETA * L2)/expSum
		if arm == 1:
			regret += (x1[i] - x1[i])
		else:
			regret += (x1[i] - x2[i])
		regretEXP3[n][i] = regret

def trainEGreedy(x1,x2,epsilon,n):
	global regretEGreedy
	global regretEGreedy2
	global optimalArmsEGreedy2
	global optimalArmsEGreedy
	mean1 = 0.0
	mean2 = 0.0
	count1 = 0
	count2 = 0
	regret = 0
	if epsilon == 0.1:
		for i in xrange(len(x1)):
			if i % 10 == 0:
				arm = random.randint(1,2)
			else:
				arm = np.argmax([mean1,mean2]) + 1
			if arm == 1:
				count1 += 1
				mean1 = updateMean(mean1,x1[i],count1)
				# Because Arm 1 is optimal, it is selected
				optimalArmsEGreedy[n][i] += 1
			else:
				count2 += 1
				mean2 = updateMean(mean2,x2[i],count2)
			regret += (x1[i]-eval('x'+str(arm)+'[i]'))
			regretEGreedy[n][i] += regret

	elif epsilon == 0.01:
		for i in xrange(len(x1)):
			if i % 100 == 0:
				arm = random.randint(1,2)
			else:
				arm = np.argmax([mean1,mean2]) + 1
			if arm == 1:
				count1 += 1
				mean1 = updateMean(mean1,x1[i],count1)
				# Because Arm 1 is optimal, it is selected
				optimalArmsEGreedy2[n][i] += 1
			else:
				count2 += 1
				mean2 = updateMean(mean2,x2[i],count2)
			regret += (x1[i]-eval('x'+str(arm)+'[i]'))
			regretEGreedy2[n][i] += regret

def averageAndPlotLists(i):
	global regretUCB
	global regretEXP3
	global regretEGreedy
	global regretEGreedy2
	global optimalArmsUCB
	global optimalArmsEXP3
	global optimalArmsEGreedy
	global optimalArmsEGreedy2
	probVal1, probVal2 = probArm1[i],probArm2[i]
	algoList = ["EXP3","UCB","EGreedy","EGreedy2"]
	meanOptArmEXP3 = list()
	meanOptArmUCB = list()
	meanOptArmEGreedy = list()
	meanOptArmEGreedy2 = list()
	stdErrOptArmEXP3 = list()
	stdErrOptArmUCB = list()
	stdErrOptArmEGreedy = list()
	stdErrOptArmEGreedy2 = list()
	meanRegretEXP3 = list()
	meanRegretUCB = list()
	meanRegretEGreedy = list()
	meanRegretEGreedy2 = list()
	stdErrRegretEXP3 = list()
	stdErrRegretUCB = list()
	stdErrRegretEGreedy = list()
	stdErrRegretEGreedy2 = list()
	for algo in algoList:
		l1 = np.mean(eval("optimalArms"+algo),axis=0)
		l2 = stats.sem(eval("optimalArms"+algo),axis=0)
		l3 = np.mean(eval("regret"+algo),axis=0)
		l4 = stats.sem(eval("regret"+algo),axis=0)
		if algo not in ["EGreedy2","EGreedy"]:
			plt.plot(range(1,NUM_TRIALS+1),l1,c="red",label=algo)
			plt.xlabel("Number of Trials")
			plt.ylabel("Number of Times Optimal Arms Played")
			plt.legend(loc="best")
			plt.suptitle("Number of Times Optimal Arms Played v/s Number of Trials for "+str(probVal1)+" & "+str(probVal2))
			plt.show()
			plt.plot(range(1,NUM_TRIALS+1),l3,c="red",label=algo)
			plt.xlabel("Number of Trials")
			plt.ylabel("Regret")
			plt.legend(loc="best")
			plt.suptitle("Regret v/s Number of Trials for "+str(probVal1)+" & "+str(probVal2))
			plt.show()
			plt.errorbar(range(1,NUM_TRIALS+1),l1,yerr=l2)
			plt.xlabel("Number of Trials")
			plt.ylabel("Number of Times Optimal Arms Played")
			plt.suptitle("Number of Times Optimal Arms Played v/s Number of Trials for "+str(probVal1)+" & "+str(probVal2)+" (with ErrorBars)")
			plt.show()
		if algo == "EGreedy":
			plt.plot(range(1,NUM_TRIALS+1),l1,c="red",label="Epsilon Greedy E = 0.1")
			plt.xlabel("Number of Trials")
			plt.ylabel("Number of Times Optimal Arms Played")
			plt.legend(loc="best")
			plt.suptitle("Number of Times Optimal Arms Played v/s Number of Trials for "+str(probVal1)+" & "+str(probVal2))
			plt.show()
			plt.plot(range(1,NUM_TRIALS+1),l3,c="red",label="Epsilon Greedy E = 0.1")
			plt.xlabel("Number of Trials")
			plt.ylabel("Regret")
			plt.legend(loc="best")
			plt.suptitle("Regret v/s Number of Trials for "+str(probVal1)+" & "+str(probVal2))
			plt.show()
			plt.errorbar(range(1,NUM_TRIALS+1),l1,yerr=l2)
			plt.xlabel("Number of Trials")
			plt.ylabel("Number of Times Optimal Arms Played")
			plt.suptitle("Number of Times Optimal Arms Played v/s Number of Trials for "+str(probVal1)+" & "+str(probVal2)+" (with ErrorBars)")
			plt.show()
		if algo == "EGreedy2":
			plt.plot(range(1,NUM_TRIALS+1),l1,c="red",label="Epsilon Greedy E = 0.01")
			plt.xlabel("Number of Trials")
			plt.ylabel("Number of Times Optimal Arms Played")
			plt.legend(loc="best")
			plt.suptitle("Number of Times Optimal Arms Played v/s Number of Trials for "+str(probVal1)+" & "+str(probVal2))
			plt.show()
			plt.plot(range(1,NUM_TRIALS+1),l3,c="red",label="Epsilon Greedy E = 0.01")
			plt.xlabel("Number of Trials")
			plt.ylabel("Regret")
			plt.legend(loc="best")
			plt.suptitle("Regeret v/s Number of Trials for "+str(probVal1)+" & "+str(probVal2))
			plt.show()
			plt.errorbar(range(1,NUM_TRIALS+1),l1,yerr=l2)
			plt.xlabel("Number of Trials")
			plt.ylabel("Number of Times Optimal Arms Played")
			plt.suptitle("Number of Times Optimal Arms Played v/s Number of Trials for "+str(probVal1)+" & "+str(probVal2)+" (with ErrorBars)")
			plt.show()
		if algo == "UCB":
			meanOptArmUCB = l1
			stdErrOptArmUCB = l2
			meanRegretUCB = l3
			stdErrRegretUCB = l4
		elif algo == "EXP3":
			meanOptArmEXP3 = l1
			stdErrOptArmEXP3 = l2
			meanRegretEXP3 = l3
			stdErrRegretEXP3 = l4
		elif algo == "EGreedy":
			meanOptArmEGreedy = l1
			stdErrOptArmEGreedy = l2
			meanRegretEGreedy = l3
			stdErrRegretEGreedy = l4
		elif algo == "EGreedy2":
			meanOptArmEGreedy2 = l1
			stdErrOptArmEGreedy2 = l2
			meanRegretEGreedy2 = l3
			stdErrRegretEGreedy2 = l4
	# Plotting the Lists Together for Comparision
	plt.plot(range(1,NUM_TRIALS+1),meanOptArmUCB,c="red",label="UCB")
	plt.plot(range(1,NUM_TRIALS+1),meanOptArmEGreedy,c="blue",label="E Greedy 0.1")
	plt.plot(range(1,NUM_TRIALS+1),meanOptArmEGreedy2,c="green",label="E Greedy 0.01")
	plt.plot(range(1,NUM_TRIALS+1),meanOptArmEXP3,c="yellow",label="EXP3")
	plt.xlabel("Number of Trials")
	plt.ylabel("Number of Times Optimal Arms Played")
	plt.legend(loc="best")
	plt.suptitle("Comparision of Optimal Arms for "+str(probVal1)+" & "+str(probVal2))
	plt.show()
	plt.plot(range(1,NUM_TRIALS+1),meanRegretUCB,c="red",label="UCB")
	plt.plot(range(1,NUM_TRIALS+1),meanRegretEGreedy,c="blue",label="E Greedy 0.1")
	plt.plot(range(1,NUM_TRIALS+1),meanRegretEGreedy2,c="green",label="E Greedy 0.01")
	plt.plot(range(1,NUM_TRIALS+1),meanRegretEXP3,c="yellow",label="EXP3")
	plt.xlabel("Number of Trials")
	plt.ylabel("Regret")
	plt.legend(loc="best")
	plt.suptitle("Comparision of Regret for "+str(probVal1)+" & "+str(probVal2))
	plt.show()

def cleanLists():
	global regretUCB
	global regretEXP3
	global regretEGreedy
	global regretEGreedy2
	global optimalArmsUCB
	global optimalArmsEXP3
	global optimalArmsEGreedy
	global optimalArmsEGreedy2
	regretUCB = np.array([[0.0] * NUM_TRIALS] * N)
	regretEXP3 = np.array([[0.0] * NUM_TRIALS] * N)
	regretEGreedy = np.array([[0.0] * NUM_TRIALS] * N)
	regretEGreedy2 = np.array([[0.0] * NUM_TRIALS] * N)
	optimalArmsUCB = np.array([[0.0] * NUM_TRIALS] * N)
	optimalArmsEXP3 = np.array([[0.0] * NUM_TRIALS] * N)
	optimalArmsEGreedy = np.array([[0.0] * NUM_TRIALS] * N)
	optimalArmsEGreedy2 = np.array([[0.0] * NUM_TRIALS] * N)

for i in xrange(len(probArm1)):
	print "Probabilities: ",probArm1[i],probArm2[i]
	for n in xrange(N):
		# Generate data
		if n % 10 == 0:
			print "\t\tExperiment Done for ",n," times."
		x1 = bernoulli.rvs(probArm1[i],size=NUM_TRIALS)
		x2 = bernoulli.rvs(probArm2[i],size=NUM_TRIALS)
		trainUCB(x1,x2,n)
		trainEXP3(x1,x2,n)
		trainEGreedy(x1,x2,0.1,n)
		trainEGreedy(x1,x2,0.01,n)
	averageAndPlotLists(i)
	cleanLists()