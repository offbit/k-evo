import numpy as np
import random
import networkx as nx
import copy

# type of nodes
nodetypes = ['sum', 'concat', 'multiply']

# type of edges
edgetype = ['fc', 'identity']

MODEL_SPACE = {}
MODEL_SPACE['fc'] = {
    "nb_units": [[32, 1024], 'int'],
    "activation": [["elu", "relu", "linear", "tanh", "sigmoid"], 'list'],
    "dropout": [[0.0, 0.8], 'float']
}

MODEL_SPACE['optimizer'] = {
    'lr': [ [0.0001, 0.1], 'float'],
    'algorithm': [['sgd', 'adam', 'adadelta', 'rmsprop'], 'list']
}

def sample_param(opts, key):
    
    if opts[key][-1] == 'int':
        min_ = opts[key][0][0]
        max_ = opts[key][0][1]
        return np.random.randint(min_, max_)

    elif opts[key][-1] == 'list':
        # "activation":[["elu", "relu", "linear", "tanh", "sigmoid"],'list'],
        return random.choice(opts[key][0])

    elif opts[key][-1] == 'float':
        min_ = opts[key][0][0]
        max_ = opts[key][0][1]
        return (np.random.rand() * (max_ - min_)) + min_
    return None


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
        edge_params = self.random_edge_params(edge_type='fc')
        edge = Edge(len(self.edges) + 1, inputnode_id,
                    outputnode_id, edge_params)
        self.add_edge(edge, update_graph=True)

        print(self.graph.nodes())
        self.opt = {}
        for k in MODEL_SPACE['optimizer']:
            self.opt[k] = sample_param(MODEL_SPACE['optimizer'], k)
        

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

        new_edge_params = self.random_edge_params(edge_type='fc')
        new_edge = Edge(len(self.edges) + 1,
                        new_node.nodeid, out_node, params=new_edge_params)
        self.add_edge(new_edge)

    def random_edge_params(self, edge_type=None):
        p = dict()
        if edge_type is None:
            if random.random() < 0.65:
                p['edgetype'] = 'fc'
            else:
                p['edgetype'] = 'identity'
        else:
            assert edge_type in ['fc', 'identity'], "Invalid edge type"
            p['edgetype'] = edge_type

        if p['edgetype'] == 'fc':
            for k in MODEL_SPACE['fc'].keys():
                p[k] = sample_param(MODEL_SPACE['fc'], k)
        
        return p

    def add_random_edge(self, edgetype=None):
        ret = -1
        while (ret == -1):
            selected_nodes = random.sample(self.nodes, 2)
            edge_params = self.random_edge_params(edgetype)
            e = Edge(len(self.edges) + 1, selected_nodes[0], selected_nodes[1], params=edge_params)

            ret = self.add_edge(e)

    def split_random_edge(self):
        edge_to_split = random.choice(self.edge_ids())
        self.split_edge(edge_to_split)

    def mutate_edge(self, edgeid):
        assert edgeid in self.edges.keys(), 'Invalid edge id'
        params = self.edges[edgeid].params
        alter_p = 0.1

        if random.random() < 0.1:
            params['edgetype'] = random.choice(['fc', 'identity'])
        
        if params['edgetype'] == 'fc':
            for key in MODEL_SPACE['fc'].keys():
                if random.random() < alter_p:
                    params[key] = sample_param(MODEL_SPACE['fc'], key)

        self.edges[edgeid].params = params
    
    def mutate_optimizer(self):
        opt = self.opt
        for k in MODEL_SPACE['optimizer']:
            opt[k] = sample_param(MODEL_SPACE['optimizer'], k)
        self.opt

    def mutate_net(self):
        add_edge_p = .2
        split_edge_p = .2
        mutate_edge = .4
        mutate_opt = .2

        choices = ['add_edge', 'split_edge', 'mutate_edge', 'mutate_opt']
        probas = [add_edge_p, split_edge_p, mutate_edge, mutate_opt]
        mutation = np.random.choice(choices, 1, p=probas)[0]
        print('mutation {}'.format(mutation))
        if mutation =='add_edge':
            self.add_random_edge()
        elif mutation == 'split_edge':
            self.split_random_edge()
        elif mutation == 'mutate_edge':            
            e = random.choice(self.edges.keys())
            print('edge e', e)
            self.mutate_edge(e)
        elif mutation == 'mutate_opt':
            self.mutate_optimizer()
        return self

def random_net(netid, input_dim, output_dim, num_mutations, classifier=True):

    m = NetModule(netid, input_dim=input_dim,
                  output_dim=output_dim, has_softmax=classifier)
    edges = m.edges.keys()
    m.split_edge(edges[-1])
    n = np.random.randint(1,num_mutations)
    for i in range(n):
        if np.random.random() < 0.5:
            m.add_random_edge()
        else:
            m.split_random_edge()
    
    return m


if __name__ == '__main__':

    m = random_net('33', 28 * 28, 10, 5, True)
    print "edge print out"
    print "=============="
    for e in m.edge_ids():
        # print e, ":", m.edges[e].in_node, "->", m.edges[e].out_node, 'type:',  m.edges[e].params['edgetype']
        print(m.edges[e])
    print(m.opt)
    m.mutate_net()
    print(m.opt)
