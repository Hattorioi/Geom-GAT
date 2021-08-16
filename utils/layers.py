import numpy as np
import tensorflow as tf
import copy

conv1d = tf.layers.conv1d

def attn_head(seq, out_sz, bias_mat, activation, in_drop=0.0, coef_drop=0.0, residual=False):
    with tf.name_scope('side_attn'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)
        # shape of seq = {h}: 1 x N x Q

        seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False) # shape of W: Q' x Q, shape of seq_fts = {W h}: 1 x N x Q'

        # simplest self-attention possible
        # a = a1 || a2 = (a1 ; a2) \in R^{2Q'}
        f_1 = tf.layers.conv1d(seq_fts, 1, 1) # shape of f_1 = {a1^T W h}: 1 x N x 1
        f_2 = tf.layers.conv1d(seq_fts, 1, 1) # shape of f_2 = {a2^T W h}: 1 x N x 1
        # In paper: a^T (W h_i || W h_j) = (a1^T , a2^T) (W h_i ; W h_j) = a1^T W h_i + a2^T W h_j
        # So, here logits[0,i,j] = a1^T W h_i + a2^T W h_j
        logits = f_1 + tf.transpose(f_2, [0, 2, 1]) # N x 1 + 1 x N → N x N (broadcast)
        # The value of non neighbor elements become negative infinity - 1e9 when plus bias_mat,
        # and after the exponent operation, the value becomes nearly 0
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)

        if coef_drop != 0.0:
            coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        vals = tf.matmul(coefs, seq_fts) # shape: N x Q'
        ret = tf.contrib.layers.bias_add(vals)

        # residual connection
        if residual:
            if seq.shape[-1] != ret.shape[-1]: # Q != Q'
                ret = ret + conv1d(seq, ret.shape[-1], 1) # activation
            else:
                ret = ret + seq

        return activation(ret), seq_fts  # activation

def attn_head_angle(seq, out_sz, bias_mat, angle_vertex, activation, in_drop=0.0, coef_drop=0.0, residual=False):
    with tf.name_scope('angle_attn'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)
        # shape of seq = {h}: 1 x N x Q

        seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False) # shape of H: Q' x Q, shape of seq_fts = {H h}: 1 x N x Q'

        # simplest self-attention possible
        # a = a1 || a2 = (a1 ; a2) \in R^{2Q'}
        f_1 = tf.layers.conv1d(seq_fts, 1, 1) # shape of f_1 = {a1^T H h}: 1 x N x 1
        f_2 = tf.layers.conv1d(seq_fts, 1, 1) # shape of f_2 = {a2^T H h}: 1 x N x 1
        # In paper: a^T (H h_i || H h_j) = (a1^T , a2^T) (H h_i ; H h_j) = a1^T H h_i + a2^T H h_j
        # So, here logits[0,i,j] = a1^T H h_i + a2^T H h_j
        logits = f_1 + tf.transpose(f_2, [0, 2, 1]) # N x 1 + 1 x N → N x N (broadcast)

        # In paper: a^T [H (h_i - h_k) || H (h_j - h_k)] = (a1^T , a2^T) [H (h_i - h_k) ; H (h_j - h_k)]
        #         = a1^T H (h_i - h_k) + a2^T H (h_j - h_k) = (a1^T H h_i + a2^T H h_j) - (a1^T H h_k + a2^T H h_k)
        # So, here logits[0,i,j] -= a1^T H h_k + a2^T H h_k
        # The following operation is equivalent to: logits[0, i, j] -= f_1[0, k, 0] + f_2[0, k, 0]
        # But a direct assignment may cause TypeError: 'Tensor' object does not support item assignment
        angle_coo_matrix = angle_vertex.tocoo() # angle_vertex stores the vertex index k of ∠ikj, converted to sparse matrix in COO format here
        indices = np.stack((angle_coo_matrix.row, angle_coo_matrix.col), axis=1) # index of (i,j)
        k = np.array(angle_coo_matrix.data - 1, dtype=np.int32) # value of k
        h_k1 = tf.gather(f_1[0], k) # shape of {a1^T H h_k}: 1 x 1 x 1
        values1 = tf.squeeze(h_k1)
        logits_add1 = tf.sparse_tensor_to_dense(tf.SparseTensor(indices=indices, values=values1, dense_shape=angle_coo_matrix.shape))
        h_k2 = tf.gather(f_2[0], k) # shape of {a2^T H h_k}: 1 x 1 x 1
        values2 = tf.squeeze(h_k2)
        logits_add2 = tf.sparse_tensor_to_dense(tf.SparseTensor(indices=indices, values=values2, dense_shape=angle_coo_matrix.shape))
        logits_add = tf.expand_dims(logits_add1 + logits_add2, axis=0)
        logits -= logits_add

        # The value of non neighbor elements become negative infinity - 1e9 when plus bias_mat,
        # and after the exponent operation, the value becomes nearly 0
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)

        if coef_drop != 0.0:
            coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        vals = tf.matmul(coefs, seq_fts) # shape: N x Q'
        ret = tf.contrib.layers.bias_add(vals)

        # residual connection
        if residual:
            if seq.shape[-1] != ret.shape[-1]: # Q != Q'
                ret = ret + conv1d(seq, ret.shape[-1], 1) # activation
            else:
                ret = ret + seq

        return activation(ret), seq_fts  # activation

# Experimental sparse attention head (for running on datasets such as Pubmed)
# N.B. Because of limitations of current TF implementation, will work _only_ if batch_size = 1!
def sp_attn_head(seq, out_sz, adj_mat, activation, nb_nodes, in_drop=0.0, coef_drop=0.0, residual=False):
    with tf.name_scope('sp_side_attn'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)
        # shape of seq = {h}: 1 x N x Q

        seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False) # shape of W: Q' x Q, shape of seq_fts = {W h}: 1 x N x Q'

        # simplest self-attention possible
        f_1 = tf.layers.conv1d(seq_fts, 1, 1) # shape of f_1 = {a1^T W h}: 1 x N x 1
        f_2 = tf.layers.conv1d(seq_fts, 1, 1) # shape of f_2 = {a2^T W h}: 1 x N x 1
        
        f_1 = tf.reshape(f_1, (nb_nodes, 1)) # omit the dimension of batch size, shape: N x 1
        f_2 = tf.reshape(f_2, (nb_nodes, 1))

        # Note that adj_mat is computed by process.py\preprocess_adj_bias() and used as a sparse tensor
        f_1 = adj_mat * f_1 # shape of f_1 = {(a1^T W h) ⊙ A}: N x N, note that (a1^T W h) is broadcasted before \odot operaion
        f_2 = adj_mat * tf.transpose(f_2, [1,0])
        logits = tf.sparse_add(f_1, f_2)
        # The above three steps are equivalent to: logits[i,j] = (a1^T W h_i + a2^T W h_j) ⊙ A = (a1^T W h_i) ⊙ A + (a2^T W h_j) ⊙ A
        # Take the distributive property first (⊙ A), in order to make use of the sparsity to speed up the operation

        lrelu = tf.SparseTensor(indices=logits.indices, 
                values=tf.nn.leaky_relu(logits.values), 
                dense_shape=logits.dense_shape)
        coefs = tf.sparse_softmax(lrelu)

        if coef_drop != 0.0:
            coefs = tf.SparseTensor(indices=coefs.indices,
                    values=tf.nn.dropout(coefs.values, 1.0 - coef_drop),
                    dense_shape=coefs.dense_shape)
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        # As tf.sparse_tensor_dense_matmul expects its arguments to have rank-2,
        # here we make an assumption that our input is of batch size 1, and reshape appropriately.
        # The method will fail in all other cases!
        coefs = tf.sparse_reshape(coefs, [nb_nodes, nb_nodes])
        seq_fts = tf.squeeze(seq_fts)
        vals = tf.sparse_tensor_dense_matmul(coefs, seq_fts)
        vals = tf.expand_dims(vals, axis=0)
        vals.set_shape([1, nb_nodes, out_sz])
        ret = tf.contrib.layers.bias_add(vals)

        # residual connection
        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + conv1d(seq, ret.shape[-1], 1) # activation
            else:
                ret = ret + seq

        return activation(ret), tf.expand_dims(seq_fts, axis=0)  # activation

def sp_attn_head_angle(seq, out_sz, adj_mat, angle_vertex, activation, nb_nodes, in_drop=0.0, coef_drop=0.0, residual=False):
    with tf.name_scope('sp_angle_attn'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)
        # shape of seq = {h}: 1 x N x Q

        seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False) # shape of H: Q' x Q, shape of seq_fts = {H h}: 1 x N x Q'

        # simplest self-attention possible
        f_1 = tf.layers.conv1d(seq_fts, 1, 1) # shape of f_1 = {a1^T H h}: 1 x N x 1
        f_2 = tf.layers.conv1d(seq_fts, 1, 1) # shape of f_2 = {a2^T H h}: 1 x N x 1

        # The following operation is equivalent to: logits_add[0, i, j] -= f_1[0, k, 0] + f_2[0, k, 0]
        # Complete the calculation of logits_add before the operations reshape and ⊙ A for f_1, f_2
        angle_coo_matrix = angle_vertex.tocoo() # angle_vertex stores the vertex index k of ∠ikj, converted to sparse matrix in COO format here
        indices = np.stack((angle_coo_matrix.row, angle_coo_matrix.col), axis=1) # index of (i,j)
        k = np.array(angle_coo_matrix.data - 1, dtype=np.int32) # value of k
        h_k1 = tf.gather(f_1[0], k) # shape of {a1^T H h_k}: 1 x 1 x 1
        values1 = tf.squeeze(h_k1)
        logits_add1 = tf.SparseTensor(indices=indices, values=values1, dense_shape=angle_coo_matrix.shape)
        h_k2 = tf.gather(f_2[0], k) # shape of {a2^T H h_k}: 1 x 1 x 1
        values2 = tf.squeeze(h_k2)
        logits_add2 = tf.SparseTensor(indices=indices, values=values2, dense_shape=angle_coo_matrix.shape)
        # In TensorFlow, subtraction of sparse tensors is not supported, and multiplication of two sparse tensors is also not supported
        # So we convert a sparse tensor to a dense tensor first, and then take the negative for dense tensor
        logits_add = -tf.sparse_tensor_to_dense(adj_mat) * tf.sparse_add(logits_add1, logits_add2)
        
        f_1 = tf.reshape(f_1, (nb_nodes, 1)) # omit the dimension of batch size, shape: N x 1
        f_2 = tf.reshape(f_2, (nb_nodes, 1))

        # Note that adj_mat is computed by process.py\preprocess_adj_bias() and used as a sparse tensor
        f_1 = adj_mat * f_1 # shape of f_1 = {(a1^T W h) ⊙ A}: N x N, note that (a1^T W h) is broadcasted before \odot operaion
        f_2 = adj_mat * tf.transpose(f_2, [1,0])
        logits = tf.sparse_add(f_1, f_2)
        # The above three steps are equivalent to: logits[i,j] = (a1^T W h_i + a2^T W h_j) ⊙ A = (a1^T W h_i) ⊙ A + (a2^T W h_j) ⊙ A
        # Take the distributive property first (⊙ A), in order to make use of the sparsity to speed up the operation

        logits = tf.sparse_add(logits, logits_add)

        lrelu = tf.SparseTensor(indices=logits.indices, 
                values=tf.nn.leaky_relu(logits.values), 
                dense_shape=logits.dense_shape)
        coefs = tf.sparse_softmax(lrelu)

        if coef_drop != 0.0:
            coefs = tf.SparseTensor(indices=coefs.indices,
                    values=tf.nn.dropout(coefs.values, 1.0 - coef_drop),
                    dense_shape=coefs.dense_shape)
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        # As tf.sparse_tensor_dense_matmul expects its arguments to have rank-2,
        # here we make an assumption that our input is of batch size 1, and reshape appropriately.
        # The method will fail in all other cases!
        coefs = tf.sparse_reshape(coefs, [nb_nodes, nb_nodes])
        seq_fts = tf.squeeze(seq_fts)
        vals = tf.sparse_tensor_dense_matmul(coefs, seq_fts)
        vals = tf.expand_dims(vals, axis=0)
        vals.set_shape([1, nb_nodes, out_sz])
        ret = tf.contrib.layers.bias_add(vals)

        # residual connection
        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + conv1d(seq, ret.shape[-1], 1) # activation
            else:
                ret = ret + seq

        return activation(ret), tf.expand_dims(seq_fts, axis=0)  # activation

