import networkx as nx
import numpy as np
import random
from sklearn.model_selection import StratifiedKFold

class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a torch float tensor, one-hot representation of the tag that is used as input to neural nets
            edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor
            neighbors: list of neighbors (without self-loop)
        '''
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.neighbors = []
        self.node_features = 0
        self.adjacency = 0
        self.edge_mat = 0

        self.max_neighbor = 0


def load_data(dataset, degree_as_tag):
    '''
        dataset: name of dataset
        test_proportion: ratio of test train split
        seed: random seed for random splitting of dataset
    '''

    print('loading data')
    g_list = []
    label_dict = {} # labels (class indices) of all graphs in a dictionary format
    feat_dict = {} # features (node classes) of all graphs in a dictionary format

    with open('dataset/%s/%s.txt' % (dataset, dataset), 'r') as f:
        n_g = int(f.readline().strip()) # number of graphs
        for i in range(n_g):
            row = f.readline().strip().split()
            n, l = [int(w) for w in row] # number of nodes, label for current graph
            if not l in label_dict:
                mapped = len(label_dict)
                label_dict[l] = mapped
            g = nx.Graph()
            node_tags = [] # node classes for current graph
            node_features = [] # node features for current graph
            for j in range(n):
                g.add_node(j)
                # row: node class, number of neighbor nodes, node index list of neighbor nodes..., node features (if provided) for current node
                row = f.readline().strip().split()
                # tmp: number of neighbor nodes + 2 (row[0] and row[1]), used to determine whether node features are provided
                tmp = int(row[1]) + 2
                if tmp == len(row): # node features are not provided
                    row = [int(w) for w in row]
                    attr = None
                else:
                    row, attr = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:]])
                if not row[0] in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped
                node_tags.append(feat_dict[row[0]])

                if tmp > len(row):
                    node_features.append(attr)

                for k in range(2, len(row)):
                    g.add_edge(j, row[k])

            if node_features != []:
                node_features = np.stack(node_features)
                node_feature_flag = True
            else:
                node_features = None
                node_feature_flag = False

            assert len(g) == n

            g_list.append(S2VGraph(g, l, node_tags))

    #add labels and edge_mat       
    for g in g_list:
        g.neighbors = [[] for i in range(len(g.g))] # len(g.g) represents the number of nodes in graph g
        for i, j in g.g.edges():
            g.neighbors[i].append(j)
            g.neighbors[j].append(i)
        degree_list = []
        for i in range(len(g.g)):
            g.neighbors[i] = g.neighbors[i]
            degree_list.append(len(g.neighbors[i]))
        g.max_neighbor = max(degree_list)

        g.label = label_dict[g.label]

        edges = [list(pair) for pair in g.g.edges()]
        edges.extend([[i, j] for j, i in edges])

        g.edge_mat = np.array(edges)
        g.adjacency = nx.adjacency_matrix(g.g)

    if degree_as_tag: # using node degree as tag
        for g in g_list:
            g.node_tags = list(dict(g.g.degree).values())
    # otherwise, using node class as tag

    #Extracting unique tag labels   
    tagset = set([])
    for g in g_list:
        tagset = tagset.union(set(g.node_tags))

    tagset = list(tagset)
    tag2index = {tagset[i]:i for i in range(len(tagset))}

    for g in g_list:
        g.node_features = np.zeros((len(g.node_tags), len(tagset)))
        g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1


    print('# classes: %d' % len(label_dict))
    print('# maximum node tag: %d' % len(tagset))

    print("# data: %d" % len(g_list))

    return g_list, len(label_dict)


def sample_mask(idx, nb_graphs, nb_nodes, nb_classes, nb_nodes_list):
    """Create mask."""
    mask = np.zeros((nb_graphs, nb_nodes), dtype=np.int32)
    y = np.zeros((nb_graphs, nb_nodes, nb_classes), dtype=np.int32)
    for i in idx:
        mask[i, 0:nb_nodes_list[i]] = 1
        y[i, 0:nb_nodes_list[i]] = labels[i, 0:nb_nodes_list[i]]
    return np.array(mask, dtype=np.bool), y


def separate_data(graph_list, seed, fold_idx):
    assert 0 <= fold_idx and fold_idx < 10, "fold_idx must be from 0 to 9."
    skf = StratifiedKFold(n_splits=10, shuffle = True, random_state = seed)

    labels = [graph.label for graph in graph_list]
    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels): # skf.split(X, y)
        idx_list.append(idx)
    train_idx, test_idx = idx_list[fold_idx]

    train_graph_list = [graph_list[i] for i in train_idx]
    test_graph_list = [graph_list[i] for i in test_idx]

    return train_graph_list, test_graph_list


