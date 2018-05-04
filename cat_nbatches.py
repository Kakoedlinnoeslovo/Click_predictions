import pandas as pd
import numpy as np
# import xgboost as xgb
from catboost import CatBoostClassifier, Pool
import pickle
import gc
import time
from tqdm import tqdm


train_cols = ['timestamp', 'C1', 'C9',
	              'C10', 'C11', 'C12', 'l1', 'l2']

cat_features = [1, 2, 3, 4, 5, 6, 7]

for i,chunk in tqdm(enumerate(pd.read_csv('./data/train.csv', sep=";", chunksize=10000000))):
	model = CatBoostClassifier(
		iterations=10,
		thread_count=10,
		depth=5,
		learning_rate=0.03,
		l2_leaf_reg=3.5,
		loss_function='Logloss',
		logging_level='Verbose',
		task_type='GPU'
	)

	if i!=0:
		model.load_model('./data/model_binary_snapshot.model')

	X_train = chunk[train_cols]
	y_train = chunk['label']
	train_data = Pool(X_train, y_train, cat_features=cat_features)

	del X_train, y_train
	gc.collect()

	model.fit(train_data)
	model.save_model('./data/model_binary_snapshot.model')
	del train_data
	del model
	print("model fitted...")
	gc.collect()


model = CatBoostClassifier(
		iterations=10,
		thread_count=10,
		depth=5,
		learning_rate=0.03,
		l2_leaf_reg=3.5,
		loss_function='Logloss',
		logging_level='Verbose',
		task_type='GPU',
	)

model.load_model('./data/model_binary_snapshot.model')

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
with open('./data/result_{}.txt'.format(time.time()), 'w') as f:
    f.write('Id,Click\n')
    for i in tqdm(range(1, len(predictions)+1)):
        n += 1
        f.write("{},{}\n".format(i, predictions[i-1]))
print("result ready...", n)

