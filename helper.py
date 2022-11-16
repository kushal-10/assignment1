import matplotlib.pyplot as plt
import numpy as np

from model.naivebayes import NaiveBayes, features1, features2
from model.logreg import LogReg, featurize
from evaluation import accuracy, f_1


def train_smooth(train_data, test_data):
    # YOUR CODE HERE
    #     TODO:
    #         1) Re-train Naive Bayes while varying smoothing parameter k,
    #         then evaluate on test_data.
    #         2) Plot a graph of the accuracy and/or f-score given
    #         different values of k and save it, don't forget to include
    #         the graph for your submission.

    ######################### STUDENT SOLUTION #########################
    pass
    ####################################################################


def train_feature_eng(train_data, test_data):
    # YOUR CODE HERE
    #     TODO:
    #         1) Improve on the basic bag of words model by changing
    #         the feature list of your model. Implement at least two
    #         variants using feature1 and feature2
    ########################### STUDENT SOLUTION ########################
    pass
    #####################################################################


def train_logreg(train_data, test_data):
    # YOUR CODE HERE
    #     TODO:
    #         1) First, assign each word in the training set a unique integer index
    #         with `buildw2i()` function (in model/logreg.py, not here)
    #         2) Now that we have `buildw2i`, we want to convert the data into
    #         matrix where the element of the matrix is 1 if the corresponding
    #         word appears in a document, 0 otherwise with `featurize()` function.
    #         3) Train Logistic Regression model with the feature matrix for 10
    #         iterations with default learning rate eta and L2 regularization
    #         with parameter C=0.1.
    #         4) Evaluate the model on the test set.
    ########################### STUDENT SOLUTION ########################
    X_train, Y_train = featurize(train_data)
    X_test, Y_test = featurize(test_data)

    # X_train = X_train[0:7]
    # Y_train = Y_train[0:7]
    final_weights = LogReg(0.01, 10).train(X_train, Y_train)
    # print(final_weights)

    # Evaluation
    # X_test = X_test[0:10]
    # Y_test = Y_test[0:10]

    y_test = np.dot(X_test, final_weights)

    # print(y_test)
    # print(y_test[6])
    for i in range(0, len(y_test)):
        y_test[i] = float(np.exp(y_test[i])) / float(np.exp(y_test[i]) + 1)
    # y_test = float(np.exp(y_test)) / float(np.exp(y_test) + 1)
    #
    for i in range(0, len(y_test)):
        if y_test[i] >= 0.5:
            y_test[i] = 1
        else:
            y_test[i] = 0

    res = []
    for i in range(0, len(y_test)):
        if y_test[i] == Y_test[i][0]:
            res.append(1)
        else:
            res.append(0)

    print(res)

    return None
    #####################################################################
