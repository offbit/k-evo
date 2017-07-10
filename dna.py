import numpy as np
import random
import networkx as nx
import copy

import models

# type of nodes
nodetypes = ['sum', 'concat', 'multiply']

# type of edges
edgetype = ['fc', 'identity']
space = {}
space['fc'] = {
    "nb_units": [[32, 1024], 'int'],
    "activation": [["elu", "relu", "linear", "tanh", "sigmoid"], 'list'],
    "dropout": [[0.0, 0.8], 'float']
}


class Node(object):

    def __init__(self, nodeid, params):

        self.nodeid = str(nodeid)
        self.params = params
        self.inputs = set()
        self.outputs = set()
        self.inputs_mutable = True
        self.outputs_mutable = True
        self.params_mutable = True

    def __str__(self):
        return self.nodeid

    def get_dim(self):
        if "dim" in self.params:
            return self.params['dim']

    def is_softmax(self):
        if "softmax" in self.params:
            return self.params['softmax']


class Edge(object):

    def __init__(self, edgeid, in_node, out_node, params=None):

        self.edgeid = str(edgeid)
        self.in_node = str(in_node)
        self.out_node = str(out_node)
        self.params = params

    def __str__(self):
        txt_ = "edge:{}->{} | type: {}".format(self.in_node, self.out_node, self.params['edgetype'] )
        return txt_


class NetModule(object):

    def __init__(self, modid, input_dim=784, output_dim=10, has_softmax=True):
        self.modid = str(modid)
        self.nodes = dict()
        self.edges = dict()
        self.graph = nx.DiGraph()

        # one input node
        inputnode_id = 'in'
        node_p = {"type": "input", "dim": input_dim}
        input_node = Node(inputnode_id, node_p)
        input_node.inputs_mutable = False
        self.add_node(input_node, update_graph=False)
        # one output node
        outputnode_id = 'out'
        node_p = {"type": "output", "dim": output_dim, "softmax": has_softmax}
        output_node = Node(outputnode_id, node_p)
        output_node.outputs_mutable = False
        self.add_node(output_node, update_graph=False)
        # edge between them
        # increase the edge id
        edge_params = {"edgetype": "fc",
                       "nb_units": 512, "activation": "relu", "dropout": 0.5}
        edge = Edge(len(self.edges) + 1, inputnode_id,
                    outputnode_id, edge_params)
        self.add_edge(edge, update_graph=True)
        print(self.graph.nodes())

    def update_graph(self):
        self.graph.clear()

        for node in self.nodes:
            self.graph.add_node(node)

        for e_id in self.edges:
            e = self.edges[e_id]
            self.graph.add_edge(e.in_node, e.out_node)

    def valid_graph(self, g):
        cycles = [a for a in nx.simple_cycles(g)]
        if len(cycles) > 0:
            return False
        return True

    def add_node(self, node, update_graph=True):
        assert node.nodeid not in self.nodes
        self.nodes[node.nodeid] = node
        if update_graph:
            self.update_graph()

    def add_edge(self, edge, update_graph=True):
        assert edge.edgeid not in self.edges.keys(), "edge id already exists"
        assert edge.in_node in self.nodes.keys(), "Invalid input node"
        assert edge.out_node in self.nodes.keys(), "Invalid output node"
        # Validity Check:
        G = copy.deepcopy(self.graph)
        G.add_edge(edge.in_node, edge.out_node)

        if self.valid_graph(G) is not True:
            print('Invalid edge, not added')
            return -1

        self.nodes[edge.in_node].outputs.add(edge.edgeid)
        self.nodes[edge.out_node].inputs.add(edge.edgeid)
        self.edges[edge.edgeid] = edge

        if update_graph:
            self.update_graph()
        return 1

    def node_ids(self):
        return [n for n in self.nodes]

    def edge_ids(self):
        return [e for e in self.edges]

    def split_edge(self, edgeid):

        in_node = self.edges[edgeid].in_node
        out_node = self.edges[edgeid].out_node

        node_p = {"type": "concat"}
        new_node = Node(len(self.nodes) + 1, node_p)
        self.add_node(new_node, update_graph=False)

        self.nodes[out_node].inputs.discard(edgeid)
        self.nodes[in_node].outputs.discard(edgeid)
        old_edge_params = self.edges[edgeid].params
        self.edges.pop(edgeid)

        old_edge = Edge(edgeid, in_node, new_node.nodeid,
                        params=old_edge_params)
        self.add_edge(old_edge)

        new_edge_params = self.random_edge_params()
        new_edge = Edge(len(self.edges) + 1,
                        new_node.nodeid, out_node, params=new_edge_params)
        self.add_edge(new_edge)

    def random_edge_params(self, edge_type=None):
        p = dict()
        if edge_type is None:
            p['edgetype'] = random.choice(['fc', 'identity'])
        else:
            p['edgetype'] = edge_type

        if p['edgetype'] == 'fc':

            for k in space['fc'].keys():
                # "nb_units":[[32, 1024], 'int']
                if space['fc'][k][-1] == 'int':
                    min_ = space['fc'][k][0][0]
                    max_ = space['fc'][k][0][1]
                    p[k] = np.random.randint(min_, max_)

                elif space['fc'][k][-1] == 'list':
                    # "activation":[["elu", "relu", "linear", "tanh", "sigmoid"],'list'],
                    p[k] = random.choice(space['fc'][k][0])

                elif space['fc'][k][-1] == 'float':
                    min_ = space['fc'][k][0][0]
                    max_ = space['fc'][k][0][1]

                    p[k] = (np.random.rand() * (max_ - min_)) + min_
        return p

    def add_random_edge(self):
        ret = -1
        while (ret == -1):
            n = random.sample(self.nodes, 2)
            e_params = self.random_edge_params()
            e = Edge(len(self.edges) + 1, n[0], n[1], params=e_params)

            ret = self.add_edge(e)

    def split_random_edge(self):
        edge_to_split = random.choice(self.edge_ids())
        self.split_edge(edge_to_split)


def random_net(id, input_dim, output_dim, num_mutations, classifier=True):

    m = NetModule('123', input_dim=input_dim,
                  output_dim=output_dim, has_softmax=classifier)
    edges = m.edges.keys()
    m.split_edge(edges[-1])

    for i in range(num_mutations):
        if np.random.random() < 0.5:
            m.add_random_edge()
        else:
            m.split_random_edge()
    return m


if __name__ == '__main__':

    m = random_net('33', 28 * 28, 10, 3, True)
    print "edge print out"
    print "=============="
    for e in m.edge_ids():
        # print e, ":", m.edges[e].in_node, "->", m.edges[e].out_node, 'type:',  m.edges[e].params['edgetype']
        print(m.edges[e])

    mdd = models.build_module(m)
