import keras
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation, ELU
from keras.layers import Conv2D, MaxPooling2D, Input
from keras import backend as K
import numpy as np
from Queue import Queue

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


def build_module(module):
    
    inputs = Input(shape=(84,))
    G = {}

    for nd in module.node_ids():
        G[nd] = []
    
    G['in'] = inputs

    visited = set()
    seen_edge = set()
    Q = ['in']
    while Q:
        node = Q.pop(0)
        
        # if node not in visited:
        
        in_edges_ = module.nodes[node].inputs
        go_back = False
        for in_e in in_edges_:
            if module.edges[in_e].in_node not in visited:
                Q.append(module.edges[in_e].in_node)
                go_back = True
                break
        if go_back:
            
            continue

        visited.add(node)
        out_edges_ = module.nodes[node].outputs
        for e in out_edges_:
            if e not in seen_edge:
                ins = module.edges[e].in_node
                outs = module.edges[e].out_node
                Q.append(outs)
                # if ins is not 'in':
                    # Q.append(ins)
                seen_edge.add(e)
                x = Dense(5, name='op'+e)(G[ins])
                
                if G[outs] == []:
                    G[outs] = x
                else:
                    G[outs] = keras.layers.add([G[outs], x])
                
                print G
                
        print '->Q', Q
    model = Model(inputs, G['out'])

    model.summary()

    # plot_model(model, to_file='model.png')
    
