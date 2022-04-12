import numpy as np
import math
import pandas as pd
from matplotlib import pyplot as plt

features = pd.read_csv("data/question-2-features.csv")
labels = pd.read_csv("data/question-2-labels.csv")

def linear_regression(lstat_vals, labels):
    ones = [1] * len(lstat_vals)
    feature_array = [ones, lstat_vals]
    feature_array = np.transpose(feature_array)

    transpose_arr = np.transpose(feature_array)
    inverse = np.linalg.inv(np.dot(transpose_arr, feature_array))
    coefficients = np.dot(np.dot(inverse, transpose_arr), labels)

    predictions = []
    originals = []
    mse = 0
    for i in range(len(lstat_vals)):
        ground_truth = labels.iloc[i]
        originals.append(ground_truth)
        prediction = coefficients[0] + coefficients[1] * lstat_vals[i]
        predictions.append(prediction)
        mse += math.pow((ground_truth - prediction), 2)
    mse = mse / len(lstat_vals)

    print("Mean Squared Error: ", mse)
    plt.scatter(lstat_vals, predictions, label = "prediction")
    plt.scatter(lstat_vals, originals, label = "ground truth")
    plt.xlabel('lstat')
    plt.ylabel('house prices')
    plt.title('house price vs lstat ground truth and predictions')
    plt.legend()
    plt.savefig("linear.jpg")
    plt.show()

def polynomial_regression(lstat_vals, labels) :
    ones = [1] * len(lstat_vals)
    lstat_squares = np.power(lstat_vals, 2)
    feature_array = [ones, lstat_vals, lstat_squares]
    feature_array = np.transpose(feature_array)

    transpose_arr = np.transpose(feature_array)
    inverse = np.linalg.inv(np.dot(transpose_arr, feature_array))
    coefficients = np.dot(np.dot(inverse, transpose_arr), labels)

    predictions = []
    originals = []
    mse = 0
    for i in range(len(lstat_vals)) :
        ground_truth = labels.iloc[i]
        originals.append(ground_truth)
        prediction = coefficients[0] + coefficients[1] * lstat_vals[i] + coefficients[2] * lstat_squares[i]
        predictions.append(prediction)
        mse += math.pow((ground_truth - prediction), 2)
    mse = mse / len(lstat_vals)

    print("Mean Squared Error: ", mse)
    plt.scatter(lstat_vals, predictions, label="prediction")
    plt.scatter(lstat_vals, originals, label="ground truth")
    plt.xlabel('lstat')
    plt.ylabel('house prices')
    plt.title('house price vs lstat ground truth and predictions')
    plt.legend()
    plt.savefig("polynomial.jpg")
    plt.show()


lstat_vals = features['LSTAT'].values

print("LINEAR REGRESSION")
linear_regression(lstat_vals, labels)
print("POLYNOMIAL REGRESSION")
polynomial_regression(lstat_vals, labels)


