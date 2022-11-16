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

def main():
    (train_data, test_data) = read_hate_tweets(TWEETS_ANNO, TWEETS_TEXT)
    alpha = 1
    # train(train_data, alpha)
    # X = ["I", "DO", "BELIEVE", "LIFE", "IS", "GOOD"]
    # predict(X)
    c = [1 ,2, 3]
    np.resize(c, (3,1))
    print(np.shape(c))
    # print(r,r1)

def voc():
    X = [1,2,3]
    Y = [3,4,5]
    return X, Y



def ln(x):
    n = 10000
    return n * ((x ** (1 / n)) - 1)


def predict(x):
    global likelihood_list
    global prior
    global classes
    global V

    predict_list = []
    i = 0

    for word in x:
        print(word)

    for p in prior:
        likelihood = ln(prior[p])
        for word in x:
            likelihood *= ln(likelihood_list[i][word])
        i += 1
        predict_list.append(likelihood)

    mle = max(predict_list)
    mle_index = predict_list.index(mle)
    # print(classes[mle_index])



def train(train_data, alpha):
    global classes  # list of all classes, each a string
    count_classes = []  # Count of each class in training data
    mega_doc = []  # 2D list of mega document for each class

    # Adding new classes found from training data into list of 'classes'
    for i in range(0, len(train_data)):
        current_class = train_data[i][1]
        add_flag = 1
        for c in classes:
            if current_class == c:
                add_flag = 0

        if add_flag == 1:
            classes.append(train_data[i][1])
            temp_list = [train_data[i][1]]
            count_list = [0]
            mega_doc.append(temp_list)
            count_classes.append(count_list)

        current_index = classes.index(train_data[i][1])
        count_classes[current_index][0] += 1
        for j in range(0, len(train_data[i][0])):
            mega_doc[current_index].append(train_data[i][0][j])

    # Sorting Mega Document
    for k in range(0, len(classes)):
        del mega_doc[k][0]
        mega_doc[k].sort()

    # Creating Unique Vocabulary V
    temp_list = []
    for i in range(0, len(classes)):
        temp_list += mega_doc[i]
    temp_list.sort()

    global V
    V.append(temp_list[0])
    for k in range(1, len(temp_list)):
        if temp_list[k] == temp_list[k - 1]:
            continue
        else:
            V.append(temp_list[k])

    # Calculating priors
    global prior
    for cl in range(0, len(classes)):
        prior_temp = count_classes[cl][0] / len(train_data)
        prior[classes[cl]] = prior_temp

    # Creating word count for each class for likelihood
    vocabulary_list = []
    for cl in range(0, len(classes)):
        cnt = 0
        vocab_dict = {mega_doc[cl][0]: 1}
        for q in range(1, len(mega_doc[cl])):
            if mega_doc[cl][q] == V[cnt]:
                vocab_dict[mega_doc[cl][q]] += 1
            else:
                tempcnt = 0
                for tc in range(cnt + 1, len(V)):
                    tempcnt += 1
                    if mega_doc[cl][q] == V[tc]:
                        vocab_dict[mega_doc[cl][q]] = 1
                        break
                    else:
                        vocab_dict[V[tc]] = 0
                cnt += tempcnt

        vocabulary_list.append(vocab_dict)

    global likelihood_list
    for i in range(0, len(vocabulary_list)):
        likelihood = {}
        for key in vocabulary_list[i]:
            curr_count = vocabulary_list[i][key]
            lk = (curr_count + alpha) / (len(mega_doc[i]) + alpha * len(V))
            likelihood[key] = lk
        likelihood_list.append(likelihood)

    # print(likelihood_list[0]['!'])


if __name__ == "__main__":
    main()
