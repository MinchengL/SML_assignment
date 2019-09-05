import LoadDataSet
import numpy as np
from numpy import matlib


def logistics_function(x):
    result = 1 / (1 + np.exp(-x))
    return result


def predicted_value_calculation(matrix):
    f = []
    m, n = np.shape(matrix)
    for i in range(m):
        y = matrix[i][0].A.tolist()[0][0]
        result = logistics_function(y)
        f.append(result)
    return f


def gradient_descent(label_matrix, feature_matrix, alpha, maxCycles):
    min_squared_error = 0
    m, n = matlib.shape(feature_matrix)
    weight = matlib.ones((n, 1))
    best_weight = matlib.ones((n, 1))
    for i in range(maxCycles):
        predicted_value = predicted_value_calculation(feature_matrix*weight)
        predicted_matrix = np.mat(predicted_value).T
        error_matrix = label_matrix - predicted_matrix
        squared_error = error_matrix.T * error_matrix / m
        if i == 0:
            min_squared_error = squared_error
        else:
            if squared_error < min_squared_error:
                min_squared_error = squared_error
                best_weight = weight
        weight = weight + alpha*feature_matrix.T*error_matrix
    return best_weight


def model_training(label_list, feature_list, category_list):
    invalid_features = []
    for i in range(len(feature_list)):
        if len(feature_list[i]) != 3:
            invalid_features.append(i)
    for i in range(len(invalid_features)):
        feature_list.pop(invalid_features[i])
        label_list.pop(invalid_features[i])
    feature_matrix = np.mat(feature_list)
    weights = []
    for i in range(len(category_list)):
        train_label_list = []
        for j in range(len(label_list)):
            if label_list[j] == category_list[i]:
                train_label_list.append(1)
            else:
                train_label_list.append(0)
        train_label_matrix = np.mat(train_label_list).T
        weight = gradient_descent(train_label_matrix, feature_matrix, 0.1, 10)
        weights.append(weight)
    return weights


def logistic_regression():
    label_list, feature_list = LoadDataSet.load_dataset("resource/train_tweets.txt")
    category_list = list(set(label_list))
    weights = model_training(label_list, feature_list, category_list)
    test_label_list, test_feature_list = LoadDataSet.load_test_dataset("resource/train_tweets.txt")
    invalid_features = []
    for i in range(len(test_feature_list)):
        if len(test_feature_list[i]) != 3:
            invalid_features.append(i)
    for i in range(len(invalid_features)):
        test_feature_list.pop(invalid_features[i])
        test_label_list.pop(invalid_features[i])
    test_feature_matrix = np.matrix(test_feature_list)
    print(test_feature_matrix)
    predicted_results = []
    for i in range(len(weights)):
        predicted_result = predicted_value_calculation(test_feature_matrix*weights[i])
        predicted_results.append(predicted_result)
    results = []
    predicted_labels = []
    counter = 0
    for i in range(len(test_feature_list)):
        result = []
        for j in range(len(predicted_results)):
            result.append(predicted_results[j][i])
        results.append(result)
    for i in range(len(results)):
        compare_list = results[i]
        max_value = compare_list.index(max(compare_list))
        label = category_list[max_value]
        predicted_labels.append(label)
    for i in range(len(predicted_labels)):
        if predicted_labels[i] == test_label_list[i]:
            counter = counter + 1
    print(counter/len(test_label_list))


if __name__ == '__main__':
    logistic_regression()