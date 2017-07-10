import keras
from keras.datasets import mnist
import dna
from keras import backend as K
import keras.backend.tensorflow_backend 
import models
import tensorflow as tf
import numpy as np  
from keras.models import Model
from keras.layers import Input
# tf.reset_default_graph()

# g = tf.Graph()

# with g.as_default():

# sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
#                                     log_device_placement=True))

# keras.backend.tensorflow_backend.set_session(sess)

K.clear_session()

batch_size = 128
num_classes = 10
epochs = 2

# input image dimensions
img_rows, img_cols = 28, 28
n_dim = img_rows * img_cols

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()


x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)
input_shape = (n_dim)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

m = dna.NetModule('123', output_dim=256, has_softmax=False)

edges = m.edges.keys()
m.split_edge(edges[-1])

for i in range(4):
    if np.random.random() < 0.5:
        m.add_random_edge()
    else:
        m.split_random_edge()    

m2 = dna.NetModule('456', input_dim=256, output_dim=10)

edges = m2.edges.keys()
m2.split_edge(edges[-1])

for i in range(2):
    if np.random.random() < 0.5:
        m2.add_random_edge()
    else:
        m2.split_random_edge()    


model1 = models.build_module(m)
model2 = models.build_module(m2)

inputs1 = Input(shape=(784,))
out1 = model1(inputs1)
out2 = model2(out1)

model = Model(inputs1, out2)
model.summary()

model.compile(loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adam(),
            metrics=['accuracy'])

model.fit(x_train[::2], y_train[::2],
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
