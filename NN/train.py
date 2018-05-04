from utils import ReaderSubmitor
import numpy as np




class Network:
	def __init__(self):
		pass
	def build_model(self):
		pass
	def fit(self):
		pass
	def predict(self):
		pass

def main():
	reader = ReaderSubmitor()
	all_values = reader.get_values()
	chunk = reader.next_chunk(all_values=all_values, chunksize=100)
	X_train, y_train = next(chunk)
	matrix = X_train.todense()
	print(matrix.shape)
	print('here')


if __name__ == '__main__':
	main()
