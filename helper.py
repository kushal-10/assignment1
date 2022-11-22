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
    accuracy_values = []
    f1_values = []
    k_values = [0.1, 0.2, 1, 2]
    for kv in k_values:
        print("For k value of : " + str(kv))
        nb = NB.train(train_data, kv)
        accval = accuracy(nb, test_data)
        accuracy_values.append(accval)
        print("Acc: " + str(accval))
        f1val = f_1(nb, test_data)
        f1_values.append(f1val)
        print("F1 : " + str(f1val))

    # PLOTTING A GRAPH OF ACCURACY & F1 SCORES AGAINST DIFFERENT K VALUES
    plt.plot(k_values, accuracy_values, 'r', label='Accuracy')
    plt.title("Accuracy against different k values")
    plt.xlabel("K values")
    plt.ylabel("Accuracy")
    plt.legend(loc="upper left")
    plt.show()
    plt.plot(k_values, f1_values, 'b', label='F1 Score')
    plt.title("F1 scores against different k values")
    plt.xlabel("K values")
    plt.ylabel("F1 score")
    plt.legend(loc="upper left")
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
    train_data = features1(train_data)
    naive = NB.train(train_data)
    print("Accuracy with feature 1: ", accuracy(naive, test_data))
    print("F_1 with feature 1: ", f_1(naive, test_data))

    train_data = features2(train_data)
    naive = NB.train(train_data)
    print("Accuracy with feature 2: ", accuracy(naive, test_data))
    print("F_1 with feature 2: ", f_1(naive, test_data))
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
    X_train, Y_train = featurize(train_data, train_data)
    X_test, Y_test = featurize(test_data, train_data)

    # GETTING WEIGHTS AND BIAS VALUES AFTER TRAINING
    final_weights, final_b = LogReg(0.01, 10).train(X_train, Y_train)

    # CALCULATING WX + B FOR TEST DATA
    y_test = np.dot(X_test, final_weights) + final_b
    (r1, c1) = np.shape(X_test)
    np.reshape(y_test, (r1, 1))

    # PREDICTING A CLASS FOR TEST DATA (0 - NONOFFENSIVE, 1 - OFFENSIVE)
    for i in range(0, len(y_test)):
        y_test[i][0] = float(np.exp(y_test[i][0])) / float(np.exp(y_test[i][0]) + 1)
        if y_test[i][0] >= 0.5:
            y_test[i][0] = 1
        else:
            y_test[i][0] = 0

    # CALCULATING ACCURACY
    res = []
    true_pos = 0
    for i in range(0, len(y_test)):
        if y_test[i][0] == Y_test[i][0]:
            res.append(1)
            true_pos += 1
        else:
            res.append(0)

    print("Accuracy: " + str(float(true_pos)/len(res)))

    return None
    #####################################################################
