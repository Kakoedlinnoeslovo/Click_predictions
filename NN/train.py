from utils import ReaderSubmitor
import numpy as np
from keras.layers import Input, Dense, BatchNormalization, Activation
import keras
from keras.models import Model
from keras import regularizers
from keras import optimizers
from tqdm import tqdm
from utils import make_prediction
import argparse
from sklearn import preprocessing


class Network:
	def __init__(self):
		self.model = None

	def build_model(self, hidden_layers, shapes, reg_lam=0.001):
		x_s = list()
		x0_s = list()
		for shape in shapes:
			# its objects: movie, user or something else
			x01 = Input(shape=(shape,))
			x0_s.append(x01)
			x1 = Dense(hidden_layers[0], kernel_regularizer=regularizers.l2(reg_lam))(x01)
			x1 = BatchNormalization()(x1)
			x1 = Activation('relu')(x1)
			x_s.append(x1)

		x_seen = list()
		x_emb = list()
		for i, x_i in enumerate(x_s):
			for j, x_j in enumerate(x_s):
				if i != j and ([i, j] not in x_seen or [j, i] not in x_seen):
					x = keras.layers.dot([x_i, x_j], axes=1)
					x_seen.append([i, j])
					x_seen.append([j, i])
					x_emb.append(x)
				else:
					continue

		x_out = keras.layers.concatenate(x_emb, axis=-1)

		#x3 = keras.layers.concatenate(x0_s, axis=-1)
		#x4 = Dense(hidden_layers[1])(x3)
		#x4 = BatchNormalization()(x4)
		#x4 = Activation('relu')(x4)

		#x_out = keras.layers.concatenate([x, x4], axis=-1)
		# x = Flatten()(x)
		x = Dense(1)(x_out)
		x = Activation('sigmoid')(x)
		model = Model(inputs=x0_s, output=x)
		self.model = model
		model.summary()

	def fit(self, X_train, y_train, learning_rate=0.01, batch_size=500, niter=10):
		if self.model is None:
			print('first you need build_model()')
		else:
			self.model.compile(optimizer=optimizers.Adam(lr=learning_rate), loss='binary_crossentropy',
			                   metrics=['accuracy'])
			self.model.fit(X_train, y_train, batch_size=batch_size, epochs=niter)

	def predict(self, X_test):
		y_pred = self.model.predict(X_test)
		return y_pred


def main(nsteps, chunksize):
	print('Init model with nsteps: {}, chunksize: {} \n'.format(nsteps, chunksize))
	reader = ReaderSubmitor()
	all_values, data_len = reader.get_values()
	model = Network()

	chunk_temp = reader.next_chunk(all_values=all_values, chunksize=1)
	X_train_temp, y_train_temp = next(chunk_temp)
	shapes = list()
	for matrix in X_train_temp:
		shapes.append(matrix.shape[1])


	print('start building network')
	model.build_model(hidden_layers=[32, 16], shapes=shapes)

	print('start fitting network')
	train_step = 0
	for X_train, y_train in tqdm(reader.next_chunk(all_values=all_values,
	                                               chunksize=chunksize,
	                                               is_train=True)):
		print()
		model.fit(X_train, y_train)
		print('Train step: {}'.format(train_step))
		train_step+=1
		if train_step == nsteps:
			break

	print('Done fitting on step: {}'.format(train_step))

	print('start making prediction')
	predict_proba = list()
	for X_test, y_test in tqdm(reader.next_chunk(all_values=all_values,
	                                             chunksize=chunksize,
	                                             is_train=False)):
		predict_proba.extend(model.predict(X_test))

	print('start forming submit file')
	make_prediction(predict_proba)
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = 'set how many steps you want')
	parser.add_argument('--train_step', type=int)
	parser.add_argument('--chunksize', type=int)
	args = parser.parse_args()
	main(args.train_step, args.chunksize)
