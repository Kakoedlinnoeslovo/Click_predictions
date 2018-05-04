import pandas as pd
import gc
from tqdm import tqdm
import time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import hstack
from sklearn import preprocessing


class ReaderSubmitor:
	def __init__(self):
		self.train_path = '../data/train.csv'
		self.test_path = '../data/test/csv'
		self.train_cols = ['timestamp', 'C1', 'C2', 'C3', 'C4',
		                   'C5', 'C6', 'C7', 'C9',
		                   'C10', 'C11', 'C12', 'l1', 'l2']

		self.label_bin = preprocessing.LabelBinarizer(sparse_output=True)
		self.chunk_count = 0

	def get_values(self, nrows=1000):
		if nrows == -1:
			data = pd.read_csv(self.train_path, sep=";")
		else:
			data = pd.read_csv(self.train_path, sep=";", nrows=nrows)

		data_len = data.shape[0]
		all_values = dict()
		print('Start counting values in column')
		for column in tqdm(data.columns):
			all_values[column] = set(data[column])
		return all_values, data_len

	def prepare_one_hot(self, data_column, col_values):
		value_key = dict({value: i for i, value in enumerate(col_values)})
		one_hot_array = np.zeros((len(data_column), len(col_values)))
		for i, raw in enumerate(data_column.values):
			value = np.zeros((len(col_values),))
			temp_one = value_key[raw]
			value[temp_one] = 1
			one_hot_array[i] = value
		one_hot_csr = csr_matrix(one_hot_array)
		return one_hot_csr

	def next_chunk(self, all_values, chunksize=1000000, is_train=True, nrows = 1000):
		if is_train:
			path = self.train_path
		else:
			path  = self.test_path

		for batch in pd.read_csv(path, sep=";", chunksize=chunksize, nrows=nrows):
			X_train = batch[self.train_cols]
			y_train = batch['label']
			print('Start making chunk {}'.format(self.chunk_count))
			self.chunk_count += 1
			train_csr = list()
			for j, column in enumerate(self.train_cols):
				train_csr.append(self.prepare_one_hot(X_train[column], all_values[column]))
			yield train_csr, y_train


def make_prediction(preds_proba):
	predictions = preds_proba[:, 1]
	print("start making prediction file")
	n = 0
	sub_time = time.time()
	with open('./data/result_{}.txt'.format(sub_time), 'w') as f:
		f.write('Id,Click\n')
		for i in tqdm(range(1, len(predictions) + 1)):
			n += 1
			f.write("{},{}\n".format(i, predictions[i - 1]))
	print("result ready...", n)

	sub_pd = pd.read_csv('./data/result_{}.txt'.format(sub_time))
	sub_pd.to_csv('./data/csv/result_{}.csv'.format(sub_time), index=False)
	print("csv ready")