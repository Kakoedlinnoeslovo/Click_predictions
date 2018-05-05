import pandas as pd
import gc
from tqdm import tqdm
import time
import numpy as np
from scipy.sparse import csr_matrix
import pickle



class ReaderSubmitor:
	def __init__(self):
		self.train_path = '../data/train.csv'
		self.test_path = '../data/test.csv'
		self.train_cols = ['C1', 'C2', 'C3', 'C4','C5', 'C6', 'C7', 'C8', 'C9',
		                   'C10', 'C11', 'C12', 'l1', 'l2']

		self.chunk_count = 0


	def get_values(self):
		data = pd.read_csv(self.train_path, sep=";")
		all_values = dict()

		print('Start counting values in train column')
		for column in tqdm(self.train_cols):
			all_values[column] = set(data[column])
			print('The length of {} equal {}\n'.format(column, len(all_values[column])))
		del data
		gc.collect()

		print('Start counting values in test column')
		data = pd.read_csv(self.test_path, sep=";")
		for column in tqdm(self.train_cols):
			all_values[column].update(set(data[column]))
			print('The length of {} equal {}\n'.format(column, len(all_values[column])))

			with open('../data/{}.dump'.format(str(column)), 'wb') as fp:
				pickle.dump(all_values[column], fp)
		del data
		gc.collect()
		del all_values
		gc.collect()


	def prepare_one_hot(self, data_column, col_set):
		value_key = dict({value: i for i, value in enumerate(col_set)})
		one_hot_array = np.zeros((len(data_column), len(col_set)))
		for i, raw in enumerate(data_column.values):
			value = np.zeros((len(col_set),))
			try:
				temp_one = value_key[raw]
				value[temp_one] = 1
			except:
				print('fail on {}'.format(raw))
			one_hot_array[i] = value
		one_hot_csr = csr_matrix(one_hot_array)

		del value_key
		gc.collect()

		return one_hot_csr


	def next_chunk(self, chunksize=1000000, is_train=True):
		if is_train:
			path = self.train_path
		else:
			path  = self.test_path

		for batch in pd.read_csv(path, sep=";", chunksize=chunksize):
			X_train = batch[self.train_cols]
			y_train = batch['label']
			print('Start making chunk {}'.format(self.chunk_count))
			self.chunk_count += 1
			train_csr = list()
			for column in self.train_cols:
				with open('../data/{}.dump'.format(str(column)), 'rb') as fp:
					temp_column = pickle.load(fp)
				train_csr.append(self.prepare_one_hot(X_train[column], temp_column))
			yield train_csr, y_train


def make_prediction(predictions):
	#predictions = preds_proba[:, 1]
	print("start making prediction file")
	n = 0
	sub_time = time.time()
	with open('../data/result_{}.txt'.format(sub_time), 'w') as f:
		f.write('Id,Click\n')
		for i in tqdm(range(1, len(predictions) + 1)):
			n += 1
			f.write("{},{}\n".format(i, predictions[i - 1][0]))
	print("result ready...", n)

	sub_pd = pd.read_csv('../data/result_{}.txt'.format(sub_time))
	sub_pd.to_csv('../data/csv/result_{}.csv'.format(sub_time), index=False)
	print("csv ready")