import copy
import random
import numpy as np
import scipy.sparse as sp


def aug_mask_ftr(input_fea, drop_percent=0.2):
    node_num = input_fea.shape[0]
    mask_num = int(node_num * drop_percent)
    node_idx = [i for i in range(node_num)]
    mask_idx = random.sample(node_idx, mask_num)
    aug_fea = copy.deepcopy(input_fea)
    zeros = np.zeros_like(aug_fea[0])
    for i in mask_idx:
        aug_fea[i] = zeros
    return aug_fea


def aug_random_edge(input_adj, drop_percent=0.2):
    row_idx, col_idx = input_adj.nonzero()
    edge_list = []
    for i in range(len(row_idx)):
        edge_list.append((row_idx[i], col_idx[i]))

    single_edge_list = []
    for i in edge_list:
        single_edge_list.append(i)
        if i[0] != i[1]: edge_list.remove((i[1], i[0]))

    edge_num = int(len(row_idx) / 2)
    add_drop_num = int(edge_num * drop_percent / 2)
    aug_adj = copy.deepcopy(input_adj)

    # drop edge
    edge_idx = [i for i in range(edge_num)]
    drop_idx = random.sample(edge_idx, add_drop_num)
    for i in drop_idx:
        aug_adj[single_edge_list[i][0], single_edge_list[i][1]] = 0
        aug_adj[single_edge_list[i][1], single_edge_list[i][0]] = 0

    # add edge
    node_num = input_adj.shape[0]
    l = [(i, j) for i in range(node_num) for j in range(i)]
    add_list = random.sample(l, add_drop_num)
    for i in add_list:
        aug_adj[i[0], i[1]] = 1
        aug_adj[i[1], i[0]] = 1

    return aug_adj


def aug_drop_angle(input_fea, input_adj, input_B, input_angle, drop_percent=0.2):
    sp_coo_matrix = sp.coo_matrix(input_angle)
    angle_list = []
    for i, j, k in zip(sp_coo_matrix.row, sp_coo_matrix.col, sp_coo_matrix.data):
        angle_list.append((i, j, int(k)-1))
    # In order to avoid the sparseness of node index 0, all the "k" have been +1
    # when we construct input_angle. So we need to -1 here.

    single_angle_list = []
    for i in angle_list:
        single_angle_list.append(i)
        if i[0] != i[1]: angle_list.remove((i[1], i[0], i[2]))

    angle_num = int(len(single_angle_list))
    drop_num = int(angle_num * drop_percent)
    angle_idx = [i for i in range(angle_num)]
    drop_idx = random.sample(angle_idx, drop_num)

    # mask attribution
    aug_fea = copy.deepcopy(input_fea)
    zeros = np.zeros_like(aug_fea[0])
    for i in drop_idx: # mask the attributes of node k for ∠ikj
        aug_fea[single_angle_list[i][2]] = zeros

    # drop edge
    aug_adj = copy.deepcopy(input_adj)
    for i in drop_idx: # drop edge e_ik and e_jk for ∠ikj
        aug_adj[single_angle_list[i][0], single_angle_list[i][2]] = 0
        aug_adj[single_angle_list[i][2], single_angle_list[i][0]] = 0
        aug_adj[single_angle_list[i][1], single_angle_list[i][2]] = 0
        aug_adj[single_angle_list[i][2], single_angle_list[i][1]] = 0

    # drop angle
    aug_B = copy.deepcopy(input_B)
    for i in drop_idx: # drop virtual edge c_ij corresponding to ∠ikj
        aug_B[single_angle_list[i][0], single_angle_list[i][1]] = 0
        aug_B[single_angle_list[i][1], single_angle_list[i][0]] = 0

    aug_angle = copy.deepcopy(input_angle)
    for i in drop_idx: # remove ∠ikj
        aug_angle[single_angle_list[i][0], single_angle_list[i][1]] = 0
        aug_angle[single_angle_list[i][1], single_angle_list[i][0]] = 0
    aug_angle = sp.lil_matrix(aug_angle.todense())

    return aug_fea, aug_adj, aug_B, aug_angle


def aug_subgraph(input_fea, input_adj, drop_percent=0.2):
    node_num = input_fea.shape[0]
    sub_node_num = int(node_num * (1 - drop_percent))

    all_node_list = [i for i in range(node_num)]
    center_node_id = random.randint(0, node_num - 1)
    sub_node_list = [center_node_id]
    all_neighbor_list = []

    for i in range(sub_node_num - 1):
        neighbor_index = input_adj[sub_node_list[i]].nonzero()[0]
        all_neighbor_list += neighbor_index.tolist()
        all_neighbor_list = list(set(all_neighbor_list))
        new_neighbor_list = [n for n in all_neighbor_list if not n in sub_node_list]
        if len(new_neighbor_list) != 0:
            new_node = random.sample(new_neighbor_list, 1)[0]
            sub_node_list.append(new_node)
        else:
            break

    drop_node_list = sorted([i for i in all_node_list if not i in sub_node_list])
    aug_fea = delete_row_col(input_fea, drop_node_list, only_row=True)
    aug_adj = delete_row_col(input_adj, drop_node_list)

    return aug_fea, aug_adj


def delete_row_col(input_matrix, drop_list, only_row=False):
    remain_list = [i for i in range(input_matrix.shape[0]) if i not in drop_list]
    out = input_matrix[remain_list, :]
    if only_row:
        return out
    out = out[:, remain_list]

    return out
