import numpy as np
import random
import networkx as nx
import copy

import model

# type of nodes
nodetypes = ['sum', 'concat', 'multiply']

# type of edges
edgetype = ['fc', 'identity']
space = {}
space['fc']= {
        "nb_units":[[32, 1024], 'int'],
        "activation":[["elu", "relu", "linear", "tanh", "sigmoid"],'list'],
        "dropout":[[0.0, 0.8], 'float']
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



class Edge(object):

    def __init__(self, edgeid, edgetype, in_node, out_node, params=None):

        self.edgeid = str(edgeid)
        self.edgetype = edgetype
        self.in_node = str(in_node)
        self.out_node = str(out_node)
    
    def __str__(self):
        txt_ = "edge:{}->{}".format(self.in_node, self.out_node)
        return txt_
    
class NetModule(object):

    def __init__(self, modid):
        self.modid = str(modid)
        self.nodes = dict()
        self.edges = dict()
        self.graph = nx.DiGraph()
        
        # one input node
        inputnode_id = 'in'
        node_p = {"type":"sum"}
        input_node = Node(inputnode_id, node_p)
        input_node.inputs_mutable = False
        self.add_node(input_node, update_graph=False)
        # one output node
        outputnode_id = 'out'
        output_node = Node(outputnode_id, node_p)
        output_node.outputs_mutable = False
        self.add_node(output_node, update_graph=False)
        # edge between them
        # increase the edge id 
        edge = Edge(len(self.edges)+1, 'fc', inputnode_id, outputnode_id)
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
        assert edge.edgeid not in self.edges, "edge id already exists"
        assert edge.in_node in self.nodes.keys(), "Invalid input node"
        assert edge.out_node in self.nodes.keys(), "Invalid output node"
        # Validity Check:
        G = copy.deepcopy(self.graph)
        G.add_edge(edge.in_node, edge.out_node)
        # assert self.valid_graph(G), "invalid edge!"
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
    
    def mutate_module(self):
        pass
    
    
    def split_edge(self, edgeid, bottom_split=True):
        
        in_node = self.edges[edgeid].in_node
        out_node = self.edges[edgeid].out_node
        node_p = {"type":"sum"}
        
        # increase node id by 1
        new_node = Node(len(self.nodes)+1 , node_p)
        self.add_node(new_node, update_graph=False)
        # new edge added on top 
        
        if bottom_split:
            self.nodes[out_node].inputs.discard(edgeid)
            new_edge =  Edge(len(self.edges)+1, 'fc', new_node.nodeid, out_node)
            self.edges[edgeid].out_node = new_node.nodeid
            self.add_edge(new_edge)
        else:
            self.nodes[in_node].outputs.discard(edgeid)
            new_edge =  Edge(len(self.edges)+1, 'fc', in_node, new_node.nodeid)
            self.edges[edgeid].in_node = new_node.nodeid
            self.add_edge(new_edge)
    
    def add_random_edge(self):
        ret = -1
        while (ret==-1):
            n = random.sample(m.nodes, 2)
            e = Edge(len(m.edges)+1, 'fc',  n[0], n[1])
            ret = m.add_edge(e)
            
    def split_random_edge(self):
        edge_to_split = random.choice(m.edge_ids())
        m.split_edge(edge_to_split, True)

if __name__ == '__main__':

    m = NetModule('123')
    edges = m.edges.keys()
    m.split_edge(edges[-1])
        
    print "nodes:", m.node_ids()

    for i in range(4):
        if np.random.random() < 0.5:
            m.add_random_edge()
        else:
            m.split_random_edge()    

    print "edge print out"
    print "=============="
    for e in m.edge_ids():
        print e, ":", m.edges[e].in_node, "->", m.edges[e].out_node
    
    mdd = model.build_module(m)