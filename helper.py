import matplotlib.pyplot as plt
import numpy as np

from evaluation import accuracy, f_1
from model.logreg import featurize, LogReg
from model.naivebayes import NaiveBayes as NB
from model.naivebayes import features1, features2
def train_smooth(train_data, test_data):
    # YOUR CODE HERE
    #     TODO:
    #         1) Re-train Naive Bayes while varying smoothing parameter k,
    #         then evaluate on test_data.
    #         2) Plot a graph of the accuracy and/or f-score given
    #         different values of k and save it, don't forget to include
    #         the graph for your submission.
    ######################### STUDENT SOLUTION #########################
    # k_values = [1, 10, 100, 1000]
    k_values = [0.001, 0.01, 0.1, 1]
    accuracy_values = []
    f1_values = []

    for kv in k_values:
        print("For k value of : " + str(kv))
        nb = NB.train(train_data, kv)
        accval = accuracy(nb, test_data)
        accuracy_values.append(accval)
        print("Acc: " + str(accval))
        f1val = f_1(nb, test_data)
        f1_values.append(f1val)
        print("F1 : " + str(f1val))

    plt.plot(k_values, accuracy_values, 'r', k_values, f1_values, 'b')
    plt.show()

    return None
    ####################################################################


def train_feature_eng(train_data, test_data):
    # YOUR CODE HERE
    #     TODO:
    #         1) Improve on the basic bag of words model by changing
    #         the feature list of your model. Implement at least two
    #         variants using feature1 and feature2
    ########################### STUDENT SOLUTION ########################
    # train_data = features1(train_data)
    # naive = NB.train(train_data)
    # print("Accuracy: ", accuracy(naive, test_data))
    # print("F_1: ", f_1(naive, test_data))

    train_data = features2(train_data)
    naive = NB.train(train_data)
    print("Accuracy: ", accuracy(naive, test_data))
    print("F_1: ", f_1(naive, test_data))
    return None
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
    # X_train, Y_train = featurize(train_data, train_data)
    # X_test, Y_test = featurize(test_data, train_data)

    with open('test_ftz.npy', 'rb') as f:
        X_train = np.load(f)
        Y_train = np.load(f)

    with open('test_ftz1.npy', 'rb') as f:
        X_test = np.load(f)
        Y_test = np.load(f)

    final_weights = LogReg(0.05, 10).train(X_train, Y_train)
    print(np.shape(final_weights))

    # Evaluation

    y_test = np.dot(X_test, final_weights)
    (r1, c1) = np.shape(X_test)
    np.reshape(y_test, (r1, 1))

    print(y_test)
    for i in range(0, len(y_test)):
        y_test[i][0] = float(np.exp(y_test[i][0])) / float(np.exp(y_test[i][0]) + 1)

    for i in range(0, len(y_test)):
        if y_test[i][0] >= 0.5:
            y_test[i][0] = 1
        else:
            y_test[i][0] = 0

    res = []
    count = 0
    for i in range(0, len(y_test)):
        if y_test[i][0] == Y_test[i][0]:
            res.append(1)
            count += 1
        else:
            res.append(0)

    print(res)
    print("Accuracy: " + str(float(count)/len(res)))

    return None
    #####################################################################
