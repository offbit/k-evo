import keras
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation, ELU
from keras.layers import Conv2D, MaxPooling2D, Input
from keras import backend as K
import numpy as np
from Queue import Queue
from keras.utils import plot_model


class CustomModel():
    def __init__(self, module):
        self.model = build_module(module)
        # self.model_info = model_info

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
    indim = module.nodes['in'].get_dim()
    inputs = Input(shape=(indim,))
    net_graph = {}

    for nd in module.node_ids():
        net_graph[nd] = []

    net_graph['in'] = inputs

    visited = set()
    seen_edge = set()
    node_q = ['in']
    while node_q:
        node = node_q.pop()
        # if node not in visited:
        in_edges_ = module.nodes[node].inputs
        go_back = False
        for in_e in in_edges_:
            if module.edges[in_e].in_node not in visited:
                node_q.append(module.edges[in_e].in_node)
                go_back = True
                continue

        if go_back:
            continue

        visited.add(node)
        out_edges_ = module.nodes[node].outputs
        for e in out_edges_:
            if e not in seen_edge:
                edge = module.edges[e]
                ins = edge.in_node
                outs = edge.out_node
                params = edge.params
                print(edge)

                if params['edgetype'] == 'fc':
                    x = Dense(params['nb_units'], name='op' + e,
                              activation=params['activation'])(net_graph[ins])
                    x = Dropout(params['dropout'])(x)
                else:
                    x = net_graph[ins]

                node_q.append(outs)

                seen_edge.add(e)
                if net_graph[outs] == []:
                    net_graph[outs] = x
                else:
                    net_graph[outs] = keras.layers.concatenate(
                        [net_graph[outs], x])

    out_dim = module.nodes['out'].get_dim()
    act = 'linear'
    if module.nodes['out'].is_softmax():
        act = 'softmax'
    outputs = Dense(out_dim, activation=act)(net_graph['out'])
    model = Model(inputs, outputs)
    model.summary()

    plot_model(model, to_file='model.png')
    return model
