import numpy as np

from utils import read_hate_tweets
from evaluation import accuracy, f_1
from model.naivebayes import NaiveBayes
from model.logreg import LogReg, featurize
from helper import train_smooth, train_feature_eng, train_logreg
import numpy as np

from tempfile import TemporaryFile

TWEETS_ANNO = './data/NAACL_SRW_2016.csv'
TWEETS_TEXT = './data/NAACL_SRW_2016_tweets.json'


MODEL_DICT = {'naive-bayes': NaiveBayes, 'logreg': LogReg}

(train_data, test_data) = read_hate_tweets(TWEETS_ANNO, TWEETS_TEXT)

# with open('test.npy', 'wb') as f:
#     np.save(f, np.array([1, 2]))
#     np.save(f, np.array([1, 3]))
# with open('test.npy', 'rb') as f:
#     a = np.load(f)
#     b = np.load(f)

# X_train, Y_train = featurize(train_data, train_data)
#
# with open('test_ftz.npy', 'wb') as f:
#     np.save(f, X_train)
#     np.save(f, Y_train)
# X_test, Y_test = featurize(test_data, train_data)
# with open('test_ftz1.npy', 'wb') as f:
#     np.save(f, X_test)
#     np.save(f, Y_test)

x = [1,2,3,4,5]
print(x[0:4])