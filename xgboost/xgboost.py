import numpy as np
from xgboost import XGBClassifier as xgb
from tqdm import tqdm
import os
import xgboost
from sklearn.metrics import log_loss
import gc


def make_sub(pred, name, dir):
    print("Making submission of {}".format(name))
    f = open(dir + name, 'w')
    f.write('Id,Click' + '\n')
    for i, prob in tqdm(enumerate(pred)):
        f.write(str(i + 1) + "," + str(prob) + '\n')
    f.close()


def shift(pred, num0, num1):
    r_factor = num1 / num0
    if pred.ndim == 2:
        prob_sampled = pred[:, 1]
    else:
        prob_sampled = pred
    prob = prob_sampled * r_factor / (1 + (r_factor - 1) * prob_sampled)
    return prob


def make_crossval( X, y, num0, num1):
    cv_size = 0.05
    cvdir = '../data/crossval_set/'
    n_folds = 5

    for i in tqdm(range(n_folds)):
        sample_nums_1 = np.random.choice(np.where(y == 1)[0], size=int(num1 * cv_size))
        sample_nums_0 = np.random.choice(np.where(y == 0)[0], size=int(num0 * cv_size))
        sample_nums = np.concatenate([sample_nums_1, sample_nums_0])

        X[sample_nums].dump(open(cvdir + str(i) + '.x', 'wb'))
        y[sample_nums].dump(open(cvdir + str(i) + '.y', 'wb'))


def cross_val(model, num0,num1,  cvdir='../data/crossval_set/'):
    n_folds = 5

    losslist = []
    for i in tqdm(range(n_folds)):
        X = np.load(open(cvdir + str(i) + '.x', 'rb'))
        y = np.load(open(cvdir + str(i) + '.y', 'rb'))
        pred = model.predict_proba(X)
        losslist.append(log_loss(y, shift(pred, num0, num1)))
    losslist = np.array(losslist)
    return np.mean(losslist), np.std(losslist)


def prepare_data():
    y = np.load(open('../data/train.y', 'rb'))[:, 0]
    X = np.load(open('../data/train.x', 'rb'))
    num1 = np.sum(y == 1)
    num0 = np.sum(y == 0)
    sample_nums_1 = np.random.choice(np.where(y == 1)[0], size=num1)
    sample_nums_0 = np.random.choice(np.where(y == 0)[0], size=num1)
    sample_nums = np.concatenate([sample_nums_1, sample_nums_0])
    X = X[sample_nums]
    y = y[sample_nums]
    X.dump(open('../data/train_neg_sample.x', 'wb'))
    y.dump(open('../data/train_neg_sample.y', 'wb'))
    return X, y, num0, num1


def make_sparse(X, y, file):
    categorical = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13}
    continual = {10, 11}

    def make_categorial(X):
        feature_dict = dict()
        c = sum([len(feature_dict[x]) for x in feature_dict if x in categorical])
        for i in tqdm(range(X.shape[1])):
            if i in categorical:
                if i not in feature_dict:
                    feature_dict[i] = {}
                for x in np.unique(X[:, i]):
                    if x not in feature_dict[i]:
                        feature_dict[i][int(x)] = str(c) + ':1 '
                        c += 1
            if i in continual:
                if i not in feature_dict:
                    feature_dict[i] = c
                    c += 1
        return feature_dict

    feature_dict = make_categorial(X)
    fsvm = open(file, 'w')
    for i in tqdm(range(y.shape[0])):
        line = str(y[i]) + ' '
        for j in range(X.shape[1]):
            x_val = X[i,j]
            if j in categorical:
                if x_val in feature_dict[j]:
                    line += feature_dict[j][x_val]
            if j in continual:
                line += str(feature_dict[j]) + ':' + str(x_val) + ' '
        fsvm.write(line + '\n')
    fsvm.close()


def main():
    X, y, num0, num1 = prepare_data()
    #model = xgb()
    #model.fit(X, y)
    #p = model.predict_proba(X)
    make_crossval(X,y, num0, num1)
    X = np.load('..data/train_neg_sample.x')
    y = np.load('..data/train_neg_sample.y')
    #cross_val(model, num0, num1)
    X_test = np.load(open('../data/test.x', 'rb'))
    make_sparse(X, np.zeros(X.shape[0]), '../data/sample.svm')
    make_sparse(X_test, np.zeros(X_test.shape[0]), '../data/test.svm')
    del X, y, X_test
    gc.collect()
    train = xgboost.DMatrix('../data/sample.svm')
    param = {'max_depth': 5, 'objective': 'binary:logistic'}
    bst = xgboost.train(param, train, 700)
    del train
    gc.collect()
    test = xgboost.DMatrix('../data/test.svm')
    pred = bst.predict(test)
    pred = shift(pred, num0, num1)
    make_sub(pred, name ='finalnn.csv', dir='../data/')
