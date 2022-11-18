import argparse

from utils import read_hate_tweets
from evaluation import accuracy, f_1
from model.naivebayes import NaiveBayes
from model.logreg import LogReg
from helper import train_smooth, train_feature_eng, train_logreg
import numpy as np
TWEETS_ANNO = './data/NAACL_SRW_2016.csv'
TWEETS_TEXT = './data/NAACL_SRW_2016_tweets.json'

likelihood_list = []
prior = {}
classes = []
V = []

class resultt:

    def __init__(self):
        self.X_vector_train = None
        self.Y_vector_train = None
        self.X_vector_test = None
        self.Y_vector_test = None

        (self.train_data, self.test_data) = read_hate_tweets(TWEETS_ANNO, TWEETS_TEXT)
        self.X_vector_train, self.Y_vector_train = self.featurize(self.train_data,self.train_data)
        self.X_vector_test, self.Y_vector_test = self.featurize(self.test_data, self.train_data)

    def Xtr(self):
        return self.X_vector_train

    def Ytr(self):
        return self.Y_vector_train

    def Xte(self):
        return self.X_vector_test

    def Yte(self):
        return self.Y_vector_test

    def buildw2i(self, vocab):
        """
        Create indexes for 'featurize()' function.

        Args:
            vocab: vocabulary constructed from the training set.

        Returns:
            Dictionaries with word as the key and index as its value.
        """
        # YOUR CODE HERE
        #################### STUDENT SOLUTION ######################
        vocab_dict = {}
        i = 0
        for v in vocab:
            vocab_dict[v] = i
            i += 1

        return vocab_dict

    def featurize(self, data, train_data):
        """
        Convert data into X and Y where X is the input and
        Y is the label.

        Args:
            data: Training or test data.
            train_data: Reference data to build vocabulary from.

        Returns:
            Matrix X and Y.
        """
        # YOUR CODE HERE
        ##################### STUDENT SOLUTION #######################
        # CREATE Vocabulary - vocab
        test_list = []
        for i in range(0, len(train_data)):
            test_list += train_data[i][0]
        test_list.sort()

        vocab = [test_list[0]]
        for j in range(1, len(test_list)):
            if test_list[j] == test_list[j - 1]:
                continue
            else:
                vocab.append(test_list[j])

        # vocab = vocab[0:5]
        # Getting Vocabulary dictionary
        vocab_dict = buildw2i(vocab)

        # Creating X and Y matrices
        X = np.zeros((len(data), len(vocab)))
        Y = np.zeros((len(data), 2))

        for i1 in range(0, len(data)):
            for word1 in data[i1][0]:
                for word2 in vocab:
                    if word1 == word2:
                        this_value = vocab_dict[word2]
                        X[i1][this_value] = 1
                        break

            if data[i1][1] == 'offensive':
                Y[i1][0] = 1
            else:
                Y[i1][1] = 1

        return X, Y

