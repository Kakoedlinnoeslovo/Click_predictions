import pandas as pd
import numpy as np
# import xgboost as xgb
from catboost import CatBoostClassifier, Pool
import pickle
import gc
import time
from tqdm import tqdm

train_data = pd.read_csv('./data/train.csv', sep=";", nrows = 15000000)
print("train_data read...")

# train_cols = ['timestamp', 'C1', 'C9',
# 	              'C10', 'C11', 'C12', 'l1', 'l2']
#
# cat_features = [1, 2, 3, 4, 5, 6, 7]

train_cols = ['timestamp', 'C1', 'C2', 'C3', 'C4',
              'C5', 'C6', 'C7', 'C9',
              'C10', 'C11', 'C12', 'l1', 'l2']

X_train = train_data[train_cols]
y_train = train_data['label']

del train_data
gc.collect()

cat_features = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13]


train_data = Pool(X_train, y_train, cat_features=cat_features)

del X_train, y_train
gc.collect()

model = CatBoostClassifier(
    iterations=60,
    thread_count=10,
    depth=5,
    learning_rate=0.03,
    l2_leaf_reg=3.5,
    loss_function='Logloss',
    logging_level='Verbose',
	task_type = 'GPU'
)
gc.collect()

model.fit(train_data)
del train_data
gc.collect()
model.save_model('./data/trained_model_10_{}'.format(time.time()))
print("model fitted...")
gc.collect()

test_data = pd.read_csv('./data/test.csv', sep=";")
print("test_data read...")

X_test = test_data[train_cols]

del test_data
gc.collect()

test = Pool(X_test, cat_features = cat_features)

del X_test
gc.collect()


preds_proba = model.predict_proba(test)
print("preds_proba done...")


predictions = preds_proba[:, 1]

print("start making prediction file")
n = 0

write_time = time.time()

with open('./data/result_{}.txt'.format(write_time), 'w') as f:
    f.write('Id,Click\n')
    for i in tqdm(range(1, len(predictions)+1)):
        n += 1
        f.write("{},{}\n".format(i, predictions[i-1]))
print("result ready...", n)

result = pd.read_csv('./data/result_{}.txt'.format(write_time))
result.to_csv('./data/result_{}.csv'.format(write_time), index=False)
print('done writing to csv')

# predictions2 = preds_proba[:, 0]
#
#
# with open('predictions2.txt', 'w') as f:
#     f.write('Id,Click\n')
#     for i in range(1, len(predictions2)+1):
#         f.write("{},{}\n".format(i, predictions2[i-1]))
#
# print("preds_proba ready...")
