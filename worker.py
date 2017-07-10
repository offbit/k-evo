from multiprocessing import Queue, Process
import cv2
import numpy as np
import os


class Scheduler:
    def __init__(self, workerids):
        self._queue = Queue()
        self.workerids = workerids
        self._results = Queue()

        self.__init_workers()

    def __init_workers(self):
        self._workers = list()
        for wid in self.workerids:
            self._workers.append(CustomWorker(wid, self._queue, self._results))


    def start(self, xlist):

        # put all of models into queue
        for model_info in xlist:
            self._queue.put(model_info)

        #add a None into queue to indicate the end of task
        self._queue.put(None)

        #start the workers
        for worker in self._workers:
            worker.start()

        # wait all fo workers finish
        for worker in self._workers:
            worker.join()
        print("All workers are done")
        returns = []
        networks = []
        for i in range(len(xlist)):
            score, net = self._results.get()
            returns.append(score)
            networks.append(net)

        return networks, returns

class CustomWorker(Process):
    def __init__(self, gpuid, queue, results):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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


        x_train = x_train.reshape(x_train.shape[0],-1)
        self.x_test = self.x_test.reshape(self.x_test.shape[0],-1)

        x_train = x_train.astype('float32')
        self.x_test = self.x_test.astype('float32')
        x_train /= 255
        self.x_test /= 255
        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(self.x_test.shape[0], 'test samples')

        idxs = np.arange(x_train.shape[0])
        np.random.shuffle(idxs)
        num_examples = 12000
        self.x_train = x_train[idxs][:num_examples]
        self.y_train = y_train[idxs][:num_examples]
        

        # convert class vectors to binary class matrices
        self.y_train = to_categorical(self.y_train, num_classes)
        self.y_test = to_categorical(self.y_test, num_classes)

    def run(self):
        #set enviornment
        # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(self._gpuid)
        import models
        import keras.backend.tensorflow_backend as K
        import tensorflow as tf
        
        K.clear_session()
        tf_config = tf.ConfigProto()
        # this needs to be set to 1.0 for local usage
        tf_config.gpu_options.per_process_gpu_memory_fraction = 0.45
        tf_config.allow_soft_placement = True
        tf_config.gpu_options.allow_growth = True
        K.set_session(tf.Session(config=tf_config))
        
        while True:
            net = self._queue.get()
            if net == None:
                self._queue.put(None)
                break
            # net = net_builder.randomize_network(bounded=False)
            xnet  = models.CustomModel(net)
            score = xnet.train(self.x_train, self.y_train, self.x_test, self.y_test)
            
            print 'woker', self._gpuid, ' score', score[1]
            del xnet
            self._results.put((score[1], net))
        print 'Net done ', self._gpuid

