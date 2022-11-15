import argparse

from utils import read_hate_tweets
from evaluation import accuracy, f_1
from model.naivebayes import NaiveBayes
from model.logreg import LogReg
from helper import train_smooth, train_feature_eng, train_logreg

TWEETS_ANNO = './data/NAACL_SRW_2016.csv'
TWEETS_TEXT = './data/NAACL_SRW_2016_tweets.json'


# MODEL_DICT = {'naive-bayes': NaiveBayes, 'logreg': LogReg}


def main():
    parser = argparse.ArgumentParser(
        description='Train naive bayes or logistic regression'
    )

    # parser.add_argument(
    #     '--model', dest='model',
    #     choices=['naive-bayes', 'logreg'],
    #     help='{naive-bayes, logreg}', type=str,
    #     required=True
    # )
    #
    # parser.add_argument(
    #     '--test_smooth', dest='test_smooth',
    #     help='Train and test Naive Bayes with varying smoothing parameter k',
    #     action='store_true'
    # )
    #
    # parser.add_argument(
    #     '--feature_eng', dest='feature_eng',
    #     help='Train and test Naive Bayes with different feature types',
    #     action='store_true'
    # )

    args = parser.parse_args()

    (train_data, test_data) = read_hate_tweets(TWEETS_ANNO, TWEETS_TEXT)
    k = 1
    # model = MODEL_DICT[args.model]

    # MY TEST CODE ####################################################
    classes = []  # list of all classes, each a string
    count_classes = []  # Count of each class in training data
    mega_doc = []  # Combination of all words (incl. repetitive words) in each indexed list of resp. class

    # Getting total no. of classes
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

        # for i in range(0, len(train_data)):
        current_index = classes.index(train_data[i][1])
        count_classes[current_index][0] += 1
        for j in range(0, len(train_data[i][0])):
            mega_doc[current_index].append(train_data[i][0][j])

    # Creating Mega Document
    for k in range(0, len(classes)):
        del mega_doc[k][0]
        mega_doc[k].sort()

    # Creating Unique Vocabulary V
    temp_list = []
    for i in range(0, len(classes)):
        temp_list += mega_doc[i]
    temp_list.sort()

    V = [temp_list[0]]
    for k in range(1, len(temp_list)):
        if temp_list[k] == temp_list[k-1]:
            k += 1
        else:
            V.append(temp_list[k])

    # Calculating priors
    prior = {}
    for cl in range(0, len(classes)):
        prior_temp = count_classes[cl][0] / len(train_data)
        prior[classes[cl]] = prior_temp

    # Creating Vocabulary count for each class for likelihood
    vocabulary_list = []
    for cl in range(0, len(mega_doc)):
        vocab_dict = {mega_doc[cl][0]: 1}
        for q in range(1, len(mega_doc[cl])):
            if mega_doc[cl][q] == mega_doc[cl][q - 1]:
                vocab_dict[mega_doc[cl][q]] += 1
                q += 1
            else:
                vocab_dict[mega_doc[cl][q]] = 1
        vocabulary_list.append(vocab_dict)

    likelihood = {}
    for i in range(0, len(vocabulary_list)):
        for key in vocabulary_list[i]:
            curr_count = vocabulary_list[i][key]
            lk = (curr_count + k)/(len(mega_doc[i]) + len(V))
            likelihood[key] = lk

    print(len(vocabulary_list[0]), len(vocabulary_list[1]), len(V))

    # MY TEST CODE #####################################################


# if args.model == 'naive-bayes':
#     print("Training naive bayes classifier...")
#     nb = model.train(train_data)
#     print("Accuracy: ", accuracy(nb, test_data))
#     print("F_1: ", f_1(nb, test_data))
#
#     if args.test_smooth:
#         train_smooth(train_data, test_data)
#
#     if args.feature_eng:
#         train_feature_eng(train_data, test_data)
# else:
#     print("Training logistic regression classifier...")
#     train_logreg(train_data, test_data)


if __name__ == "__main__":
    main()
