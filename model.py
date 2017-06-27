import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, ELU
from keras.layers import Conv2D, MaxPooling2D, Input
from keras import backend as K
import numpy as np


class CustomModel():
    def __init__(self, build_info):
        
        model = Sequential()
        model.add(Flatten(input_shape=(28,28,1)))

        for i, layer_info in enumerate(build_info['layers']):
            model.add(Dense(layer_info['nb_units']['val']))
            model.add(Dropout(layer_info['dropout_rate']['val']))
            model.add(Activation(layer_info['activation']['val']))

        model.add(Dense(10, activation='softmax'))

        self.model = model
    
    def train(self, x_train, y_train, x_test, y_test):
        batch_size = 128
        epochs = 1
        self.model.compile(loss=keras.losses.categorical_crossentropy,
                    optimizer=keras.optimizers.Adadelta(),
                    metrics=['accuracy'])

        self.model.fit(x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,
                validation_data=(x_test, y_test))
        score = self.model.evaluate(x_test, y_test, verbose=0)

        return score