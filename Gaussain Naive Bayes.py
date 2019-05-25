import math
import random
import pandas as pd
import time

# Randomly split dataset into training set and test set according to split ratio
def splitDataset(dataset, splitRatio):
	trainSize = int(len(dataset) * splitRatio)
	trainSet = []
	copy = list(dataset)
	while len(trainSet) < trainSize:
		index = random.randrange(len(copy))
		trainSet.append(copy.pop(index))
	return [trainSet, copy]


# Create a dictionary to represent the data in training set according to their class(key of dict)
def separateByClass(dataset):
	separated = {}
	for i in range(len(dataset)):
		vector = dataset[i]
		if (vector[-1] not in separated):
			separated[vector[-1]] = []
		separated[vector[-1]].append(vector)
	return separated

# Calculate the mean of given numbers
def mean(numbers):
	return sum(numbers)/float(len(numbers))

# Calculate the standard deviation of given numbers (minimal value set to 0.001)
def stdev(numbers):
	avg = mean(numbers)
	variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
	return max(math.sqrt(variance),0.001)

# Calculate mean and standard deviation of each feature 
def summarize(dataset):
	summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
	del summaries[-1]
	return summaries

# Calculate all feature's mean and standard deviation of each class
def summarizeByClass(dataset):
	separated = separateByClass(dataset)
	summaries = {}
	for classValue, instances in separated.items():
		summaries[classValue] = summarize(instances)
	return summaries

# Calculate P(y) i.e.P(rumour) and P(non-rumour) in dataset
def calculate_y_probability(train_set):
    dataframe = pd.DataFrame(train_set)
    y_prob = {}
    y = dataframe.iloc[:,-1]
    P_rumour = sum(y)/len(y)
    P_nonrumour = 1.0-P_rumour
    y_prob[1] = P_rumour
    y_prob[0] = P_nonrumour
    return y_prob

# Given a feature value such as x1, estimate P(x1|y) using the value of probability density function
def calculateProbability(x, mean, stdev):
	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
	return ((1 / (math.sqrt(2*math.pi) * stdev)) * exponent)

#Calculate P(x1,x2,x3...|y)*P(y)
def calculateClassProbabilities(summaries, inputVector, y_prob):
	probabilities = {}
	for classValue, classSummaries in summaries.items():
        #probabilities[classValue] = math.log(y_prob[classValue])
		probabilities[classValue] = y_prob[classValue]
		for i in range(len(classSummaries)):
			mean, stdev = classSummaries[i]
			x = inputVector[i]
            #probabilities[classValue] += math.log(max(calculateProbability(x, mean, stdev),0.001))
			probabilities[classValue] *= calculateProbability(x, mean, stdev)
	return probabilities


# Get the class to which an input most likely belongs
def predict(summaries, inputVector, y_prob):
	probabilities = calculateClassProbabilities(summaries, inputVector, y_prob)
	bestLabel, bestProb = None, 0
	for classValue, probability in probabilities.items():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel

# Predict the classes of each example in test set
def getPredictions(summaries, testSet, y_prob):
	predictions = []
	for i in range(len(testSet)):
		result = predict(summaries, testSet[i], y_prob)
		predictions.append(result)
	return predictions

# Calculate overall accuracy in test set
def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0

# Calculate precision recall and F1 score regarding the rumour
def precision_recall_f1score(testSet,predictions):
    actual_class_list = list(list(zip(*testSet))[-1])
    num_pre_1 = sum(predictions)
    num_act_1 = sum(actual_class_list)
    index_of_1_in_prediction = [i for i,x in enumerate(predictions) if x == 1]
    index_of_1_in_actual_list = [i for i,x in enumerate(actual_class_list) if x == 1]
    
    actual_1_in_prediected_1 = 0
    for item in index_of_1_in_prediction:
        if actual_class_list[item] == 1:
            actual_1_in_prediected_1 += 1
    precision = (actual_1_in_prediected_1 / num_pre_1) 
    
    predicted_1_in_actual_1 = 0
    for item in index_of_1_in_actual_list:
        if predictions[item] == 1:
            predicted_1_in_actual_1 += 1
    recall = (predicted_1_in_actual_1 / num_act_1)
    
    f1score = 2*precision*recall / (precision + recall)
    return (precision, recall, f1score)
    



df = pd.read_csv('dataset.csv')
df = df.iloc[: , 1:]
dataset = df.values
dataset = dataset.tolist()

    
#split dataset into 85% training set and 15% test set
splitRatio = 0.85
trainingSet, testSet = splitDataset(dataset, splitRatio)
print('Split {0} rows into train={1} and test={2} rows'.format(len(dataset), len(trainingSet), len(testSet)))

start = time.time()
#Calculate P(y) in training set
y_prob = calculate_y_probability(trainingSet)

# prepare model
summaries = summarizeByClass(trainingSet)

# test model
predictions = getPredictions(summaries, testSet, y_prob)

accuracy = getAccuracy(testSet, predictions)
precision, recall, f1score = precision_recall_f1score(testSet,predictions)
print('Accuracy: {:0.3f}%'.format(accuracy))
print('Precision: {:0.3f}'.format(precision))
print('Recall: {:0.3f}'.format(recall))
print('F1-score: {:0.3f}'.format(f1score))

end = time.time()
# print execution time
print('run time:{:.3f}s\n\n'.format(end - start))
