from multiprocessing import Queue, Process
import cv2
import numpy as np
import os
import net_builder

class CustomWorker(Process):
    def __init__(self, gpuid, queue, results):
        from keras.datasets import mnist
        from keras.utils import to_categorical
        Process.__init__(self, name='ModelProcessor')
        self._gpuid = gpuid
        self._queue = queue
        self._results = results
        # Load data on the worker

        batch_size = 128
        num_classes = 10
        epochs = 1

        # input image dimensions
        img_rows, img_cols = 28, 28

        # the data, shuffled and split between train and test sets
        (x_train, y_train), (self.x_test, self.y_test) = mnist.load_data()


        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        self.x_test = self.x_test.reshape(self.x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

        x_train = x_train.astype('float32')
        self.x_test = self.x_test.astype('float32')
        x_train /= 255
        self.x_test /= 255
        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(self.x_test.shape[0], 'test samples')

        idxs = np.arange(x_train.shape[0])
        np.random.shuffle(idxs)
        num_examples = 8000
        self.x_train = x_train[idxs][:num_examples]
        self.y_train = y_train[idxs][:num_examples]
        

        # convert class vectors to binary class matrices
        self.y_train = to_categorical(self.y_train, num_classes)
        self.y_test = to_categorical(self.y_test, num_classes)

    def run(self):
        #set enviornment
        # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(self._gpuid)
        #load models
        import model

        # print 'model init done', self._gpuid

        while True:
            net = self._queue.get()
            if net == None:
                self._queue.put(None)
                break
            # net = net_builder.randomize_network(bounded=False)
            xnet  = model.CustomModel(net)
            score = xnet.train(self.x_train, self.y_train, self.x_test, self.y_test)
            
            print 'woker', self._gpuid, ' score', score
            del xnet
            self._results.put(score)
        print 'Net done ', self._gpuid

