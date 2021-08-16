import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import copy
import tensorflow as tf


"""
 Prepare adjacency matrix by expanding up to a given neighbourhood.
 This will insert loops on every node.
 Finally, the matrix is converted to bias vectors.
 Expected shape: [graph, nodes, nodes]
"""
def adj_to_bias(adj, sizes, nhood=1):
    nb_graphs = adj.shape[0]
    mt = np.empty(adj.shape)
    for g in range(nb_graphs):
        mt[g] = np.eye(adj.shape[1])
        for _ in range(nhood):
            mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
        for i in range(sizes[g]):
            for j in range(sizes[g]):
                if mt[g][i][j] > 0.0:
                    mt[g][i][j] = 1.0
    return -1e9 * (1.0 - mt)


###############################################
# This section of code adapted from tkipf/gcn #
###############################################

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str): # {'pubmed', 'citeseer', 'cora'}
    """Load data."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    #num = len(ally) + len(ty)
    idx_test = test_idx_range.tolist()  # range(int(num*0.8), num)
    idx_train = range(len(y))  # range(0, int(num*0.6))
    idx_val = range(len(y), len(y)+500)  # range(int(num*0.6), int(num*0.8))

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    print("Shape of adj:", adj.shape)
    print("Shape of features:", features.shape)

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def standardize_data(f, train_mask):
    """Standardize feature matrix and convert to tuple representation"""
    # standardize data
    f = f.todense()
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = f[:, np.squeeze(np.array(sigma > 0))]
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = (f - mu) / sigma
    return f


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


#############################################
# This section of code is for harmonic maps #
#############################################

def normalize_b(adj2):
    """Symmetrically normalize 2-order adjacency matrix."""
    adj2 = sp.coo_matrix(adj2)
    rowsum2 = np.array(adj2.sum(1), dtype=np.float32)
    d_inv = np.power(rowsum2, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    return adj2.dot(d_mat_inv).transpose().dot(d_mat_inv).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def preprocess_b(adj):
    """Preprocessing of 2nd-order adjacency matrix for simple GCN model."""
    sp_lil_matrix = sp.lil_matrix(adj) # LIL format supports flexible indexing and slicing
    first_order = sp_lil_matrix
    second_order = first_order.dot(first_order)
    first_order = sp_lil_matrix + sp.eye(adj.shape[0])
    sp_coo_matrix = first_order.tocoo() # COO format has the attributes "row" and "col"
    # For 2nd-order adjacency matrix, set to 0 at corresponding position of the nonzero elements in 1st-order adjacency matrix
    for i,j in zip(sp_coo_matrix.row, sp_coo_matrix.col):
        if first_order[i, j] != 0:
            second_order[i, j] = 0
    second_order = second_order + sp.eye(adj.shape[0])  # self-loop
    #second_order[second_order > 0.0] = 1.0

    return second_order


def preprocess_adj_bias(adj):
    num_nodes = adj.shape[0]
    adj = adj + sp.eye(num_nodes)  # self-loop
    adj[adj > 0.0] = 1.0
    if not sp.isspmatrix_coo(adj):
        adj = adj.tocoo()
    adj = adj.astype(np.float32)
    indices = np.vstack((adj.col, adj.row)).transpose()  # This is where I made a mistake, I used (adj.row, adj.col) instead
    # return tf.SparseTensor(indices=indices, values=adj.data, dense_shape=adj.shape)
    return indices, adj.data, adj.shape


def cos_sim(vector_a, vector_b):
    num = float(np.dot(vector_a, vector_b.T))
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    if denom == 0: cos = 0.0 # there is a zero vector in vector a or b
    else: cos = num / denom
    return cos


def compute_vertex(adj, features):
    first_order = sp.lil_matrix(adj)
    B = first_order.dot(first_order)
    vertex = copy.deepcopy(B)
    sp_coo_matrix2 = B.tocoo()
    for i, j in zip(sp_coo_matrix2.row, sp_coo_matrix2.col):
        if first_order[i, j] != 0 or i == j:
            B[i, j] = 0
            vertex[i, j] = 0
        else:
            i_k = first_order.rows[i] # neighborhoods for node i
            j_k = first_order.rows[j] # neighborhoods for node j
            k_list = list(set(i_k) & set(j_k)) # common neighborhoods for nodes i and j
            max_cos = -1.0 # max cos value in common neighborhoods
            max_k = k_list[0] # corresponding node index for max cos value in common neighborhoods
            for k in k_list:
                cos = cos_sim(features[i] - features[k], features[j] - features[k])
                if cos > max_cos: # update the max cos value and the corresponding node index in common neighborhoods
                    max_cos = cos
                    max_k = k
            vertex[i, j] = max_k + 1 # avoid the sparseness of node index 0
    B = B + sp.eye(adj.shape[0])
    # note that we can't directly use .tolil(), otherwise some new zeros will also be treated as non-zero elements
    # so .todense() is necessary before converting to LIL format
    B = sp.lil_matrix(B.todense())
    vertex = sp.lil_matrix(vertex.todense())
    return B, vertex


def spring_constants(angle_vertex, features):
    sp_coo_matrix = angle_vertex.tocoo()
    K = np.zeros(sp_coo_matrix.shape, dtype=np.float64)
    count1 = 0
    count2 = 0
    for i, j, k in zip(sp_coo_matrix.row, sp_coo_matrix.col, sp_coo_matrix.data):
        cos = cos_sim(features[i] - features[k-1], features[j] - features[k-1])
        cos = np.clip(cos, -1., 1.) # numeric operation may cause out of range, so limit cos value in [-1, 1]
        theta = np.arccos(cos)
        if theta != 0.: cot = 1.0 / np.tan(theta)
        else: cot = 20. # avoid dividing by 0
        if cot >= 20: count1 += 1
        else: count2 += 1
        cot = np.clip(cot, -20., 20.) # truncate cot value in [-20, 20], the corresponding θ value in [2.86, 90]
        K[i, j] = cot
    print("Number of cot value >=20 vs. <20:", count1, count2)
    return K


def harmonic_loss(K, harmonic_maps):
    harmonic_energy = 0.0
    rowsum = np.sum(K, axis=-1)
    D = np.diag(rowsum)
    L = D - K
    L = tf.cast(L, dtype=tf.float32)
    for maps in harmonic_maps:
        F = maps[0, :, :]
        result_mat = tf.matmul(tf.matmul(tf.transpose(F), L), F)
        result = tf.trace(result_mat) / 2.0
        harmonic_energy += result
    return harmonic_energy / (len(harmonic_maps))


def sp_spring_constants(angle_vertex, features):
    sp_coo_matrix = angle_vertex.tocoo()
    K = copy.deepcopy(angle_vertex).tolil()
    count1 = 0
    count2 = 0
    for i, j, k in zip(sp_coo_matrix.row, sp_coo_matrix.col, sp_coo_matrix.data):
        cos = cos_sim(features[i] - features[k-1], features[j] - features[k-1])
        cos = np.clip(cos, -1., 1.) # numeric operation may cause out of range, so limit cos value in [-1, 1]
        theta = np.arccos(cos)
        if theta != 0.: cot = 1.0 / np.tan(theta)
        else: cot = 20. # avoid dividing by 0
        if cot >= 20: count1 += 1
        else: count2 += 1
        cot = np.clip(cot, -20., 20.) # truncate cot value in [-20, 20], the corresponding θ value in [2.86, 90]
        K[i, j] = cot
    print("Cot value >=20 vs. <20:", count1, count2)
    return K


def sp_harmonic_loss(K, harmonic_maps):
    harmonic_energy = 0.0
    # note that K.todense() returns a np.matrix format, which may cause .squeeze/flatten() unuseful, so we convert it to np.array format
    rowsum = np.sum(np.array(K.todense()), axis=-1).flatten()
    D = sp.diags(rowsum)
    L = D - K
    indices, values, shape = sparse_to_tuple(L)
    L = tf.cast(tf.SparseTensor(indices=indices, values=values, dense_shape=shape), tf.float32)
    for maps in harmonic_maps:
        F = maps[0, :, :]
        # L is a sparse tensor, F is a dense tensor
        # sparse_tensor_dense_matmul() returns a dense tensor
        result_mat = tf.matmul(tf.transpose(F), tf.sparse_tensor_dense_matmul(L, F))
        result = tf.trace(result_mat) / 2.0
        harmonic_energy += result
    return harmonic_energy / (len(harmonic_maps))
