import pandas as pd
import gc
import tqdm
import time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import hstack
from sklearn import preprocessing


class ReaderPredictor:
	def __init__(self):
		self.train_path = './data/train.csv'
		self.test_path = './data/test/csv'
		self.train_cols = ['timestamp', 'C1', 'C2', 'C3', 'C4',
		                   'C5', 'C6', 'C7', 'C9',
		                   'C10', 'C11', 'C12', 'l1', 'l2']

		self.label_bin = preprocessing.LabelBinarizer(sparse_output=True)


	def get_values(self):
		data = pd.read_csv(self.train_path, sep=";")
		all_values = dict()
		for column in data.columns:
			all_values[column] = set(data[column])
		return all_values


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

	def next_chunk(self, all_values, chunksize=1000000):
		for batch in pd.read_csv(self.train_path, sep=";", chunksize=chunksize):
			X_train = batch[self.train_cols]
			y_train = batch['label']
			train_csr = None
			for j, column in enumerate(X_train.columns):
				if j == 0:
					train_csr = self.prepare_one_hot(X_train[column], all_values[column])
				else:
					one_hot_csr = self.prepare_one_hot(X_train[column], all_values[column])
					train_csr = hstack([train_csr, one_hot_csr])
			yield train_csr, y_train


	def read_test(self):
		test_data = pd.read_csv(self.test_path, sep=";")
		X_test = test_data[self.train_cols]
		del test_data
		gc.collect()
		return X_test


def make_prediction(preds_proba):
	predictions = preds_proba[:, 1]
	print("start making prediction file")
	n = 0
	with open('./data/result_{}.txt'.format(time.time()), 'w') as f:
		f.write('Id,Click\n')
		for i in tqdm(range(1, len(predictions) + 1)):
			n += 1
			f.write("{},{}\n".format(i, predictions[i - 1]))
	print("result ready...", n)


if __name__ == "__main__":
	reader = ReaderPredictor()
	chunk = reader.next_chunk(chunksize=100)
	print(next(chunk))
	print('here')