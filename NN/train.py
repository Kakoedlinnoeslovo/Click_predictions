from utils import ReaderSubmitor
import numpy as np
from keras.layers import Input, Dense, BatchNormalization, Activation
import keras
from keras.models import Model
from keras import regularizers
from keras import optimizers
from tqdm import tqdm


class Network:
    def __init__(self):
        self.model = None

    def build_model(self, hidden_layers, X_train_shape1, reg_lam=0.001):
        x01 = Input(shape=(X_train_shape1,))
        x1 = Dense(hidden_layers[0], kernel_regularizer=regularizers.l2(reg_lam))(x01)
        x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)

        x02 = Input(shape=(X_train_shape1,))
        x2 = Dense(hidden_layers[0], kernel_regularizer=regularizers.l2(reg_lam))(x02)
        x2 = BatchNormalization()(x2)
        x2 = Activation('relu')(x2)

        x = keras.layers.dot([x1, x2], axes=1)

        x3 = keras.layers.concatenate([x01, x02], axis=-1)
        x4 = Dense(hidden_layers[1])(x3)
        x4 = BatchNormalization()(x4)
        x4 = Activation('relu')(x4)

        x_out = keras.layers.concatenate([x, x4], axis=-1)
        # x = Flatten()(x)
        x = Dense(1)(x_out)
        x = Activation('sigmoid')(x)
        model = Model(inputs=[x01, x02], output=x)
        self.model = model
        model.summary()

    def fit(self, X_train, y_train, learning_rate=0.01, batch_size=500, niter=10):
        if self.model is None:
            print('first you need build_model()')
        else:
            self.model.compile(optimizer=optimizers.Adam(lr=learning_rate), loss='binary_crossentropy',
                               metrics=['accuracy'])
            self.model.fit(X_train, y_train, batch_size=batch_size, niter=niter)

    def predict(self, X_test):
        y_pred = self.model.predict(X_test)
        return y_pred


def main():
    reader = ReaderSubmitor()
    all_values, data_len = reader.get_values(nrows=1000)
    model = Network()

    chunk_temp = reader.next_chunk(all_values=all_values, chunksize=1)
    X_train_temp, y_train_temp = next(chunk_temp)
    matrix = X_train_temp.todense()

    model.build_model(hidden_layers=[32,16], X_train_shape1=matrix.shape[1])

    for chunk in tqdm(reader.next_chunk(all_values=all_values, chunksize=100)):
        X_train, y_train = next(chunk)
        #X_train_numpy = X_train.todense()
        model.fit(X_train, y_train)


if __name__ == '__main__':
    main()
