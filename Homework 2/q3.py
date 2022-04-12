import random
import numpy as np
import math
import pandas as pd

train_features = pd.read_csv("data/question-3-features-train.csv")
train_labels = pd.read_csv("data/question-3-labels-train.csv")
test_features = pd.read_csv("data/question-3-features-test.csv")
test_labels = pd.read_csv("data/question-3-labels-test.csv")

def full_batch(features, labels, learning_rate, total_iteration):
    weights = [0] * len(features.columns)
    for iteration in range(total_iteration):
        exp = np.exp(-(np.dot(features, weights)))
        prediction = 1 / (1 + exp)
        diff = labels.iloc[:,0] - prediction
        grad = np.dot(np.transpose(features), diff)
        weights += grad * learning_rate
    return weights

def mini_batch(features, labels, learning_rate, batch_size, total_iteration):
    weights = []
    for i in range(len(features.columns)):
        rand = random.gauss(0, 0.01)
        weights.append(rand)
    for iteration in range(total_iteration):
        grad = 0
        for row in range(len(labels)):
            exp = np.exp(-(np.dot(features.iloc[row], weights)))
            prediction = 1 / (1 + exp)
            diff = labels.iloc[row,0] - prediction
            grad += np.dot(np.transpose(features.iloc[row]), diff)
            if (row % batch_size == 0):
                weights += grad * learning_rate
    return weights

def stochastic(features, labels, learning_rate, total_iteration):
    weights = []
    for i in range(len(features.columns)):
        rand = random.gauss(0, 0.01)
        weights.append(rand)
    for iteration in range(total_iteration):
        grad = 0
        for row in range(len(labels)) :
            exp = np.exp(-(np.dot(features.iloc[row], weights)))
            prediction = 1 / (1 + exp)
            diff = labels.iloc[row, 0] - prediction
            grad += np.dot(np.transpose(features.iloc[row]), diff)
            weights += grad * learning_rate
    return weights

def calculate_results(weights, test_features, test_labels):
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    accuracy = 0
    exp = np.exp(-(np.dot(test_features, weights)))
    prediction = 1 / (1 + exp)
    for i in range(len(prediction)):
        if (prediction[i] >= 0.5) :
            prediction[i] = 1
        else :
            prediction[i] = 0
        if(prediction[i] == test_labels.iloc[i,0]):
            accuracy += 1
            if(prediction[i] == 0): #true neg
                true_neg += 1
            else:
                true_pos += 1
        else:
            if (prediction[i] == 0) :  # false neg
                false_neg += 1
            else :
                false_pos += 1
    accuracy = accuracy / len(test_labels)
    print("Accuracy: ", accuracy)
    print("True pos: ", true_pos)
    print("False pos: ", false_pos)
    print("True neg: ", true_neg)
    print("False neg: ", false_neg)
    precision = 0
    recall = 0
    if (true_pos + false_pos == 0):
        print("division by zero: precision")
        print("division by zero: fdr")
    else:
        precision = true_pos / (true_pos + false_pos)
        fdr = false_pos / (true_pos + false_pos)
        print("Precision: ", precision)
        print("FDR: ", fdr)
    if (true_pos + false_neg == 0) :
        print("division by zero: recall")
    else:
        recall = true_pos / (true_pos + false_neg)
        print("Recall: ", recall)
    if (true_neg + false_neg == 0):
        print("division by zero: npv")
    else:
        npv = true_neg / (true_neg + false_neg)
        print("NPV: ", npv)
    if (false_pos + true_neg == 0):
        print("division by zero: fpr")
    else:
        fpr = false_pos / (false_pos + true_neg)
        print("FPR: ", fpr)

    if(precision + recall == 0):
        print("division by zero: f1")
    else:
        f1 = (2 * precision * recall) / (precision + recall)
        print("F1: ", f1)

    if (0.25 * precision + recall == 0) :
        print("division by zero: f2")
    else :
        f2 = (1.25 * precision * recall) / (0.25 * precision + recall)
        print("F2: ", f2)

normalized_train_features = (train_features-train_features.min())/(train_features.max()-train_features.min())
normalized_test_features = (test_features-test_features.min())/(test_features.max()-test_features.min())
normalized_train_features.insert(0, "x0", 1)
normalized_test_features.insert(0, "x0", 1)
total_iteration = 1000
print("FULL BATCH")
for lr in range(5):
    learning_rate = math.pow(10, -1 * (lr + 1))
    weights = full_batch(normalized_train_features, train_labels, learning_rate, total_iteration)
    print(weights)
    print("Learning rate: ", learning_rate)
    calculate_results(weights, normalized_test_features, test_labels)

learning_rate = math.pow(10, -2)
print("MINI BATCH")
weights = mini_batch(normalized_train_features, train_labels, learning_rate, 100, total_iteration)
print(weights)
calculate_results(weights, normalized_test_features, test_labels)
print("STOCHASTIC")
weights = stochastic(normalized_train_features, train_labels, learning_rate, total_iteration)
print(weights)
calculate_results(weights, normalized_test_features, test_labels)
