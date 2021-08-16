import time
import numpy as np
import pickle as pkl
import tensorflow as tf

from augs import aug
from augs import dgi
from models import GeomGAT
from utils import process


checkpt_file = 'pre_trained/cora/mod_cora.ckpt'

dataset = 'cora'

# augmentation params
aug_type = 'angle'
aug_percent = 0.2 # dropping or masking ratio of data augmentation

# training params
batch_size = 1
nb_epochs = 100000
patience = 100
lr = 0.005  # learning rate
l2_coef = 0.0005  # weight decay
hid_units = [8] # numbers of hidden units per each attention head in each layer
n_heads = [4, 1] # additional entry for the output layer
residual = False
nonlinearity = tf.nn.elu
lam = 0.1 # coefficient of harmonic energy
mu = 1 # coefficient of contrastive loss
model = GeomGAT

print('Dataset: ' + dataset)
print('----- Opt. hyperparams -----')
print('lr: ' + str(lr))
print('l2_coef: ' + str(l2_coef))
print('lambda: ' + str(lam))
print('mu: ' + str(mu))
print('----- Archi. hyperparams -----')
print('nb. layers: ' + str(len(hid_units)))
print('nb. units per layer: ' + str(hid_units))
print('nb. attention heads: ' + str(n_heads))
print('residual: ' + str(residual))
print('nonlinearity: ' + str(nonlinearity))
print('model: ' + str(model))

adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = process.load_data(dataset)
features, spars = process.preprocess_features(features)
B, angle_vertex = process.compute_vertex(adj, features)
K = process.spring_constants(angle_vertex, features)

nb_nodes = features.shape[0]
ft_size = features.shape[1]
nb_classes = y_train.shape[1]

features = np.array(features)
adj = np.array(adj.todense())
B = np.array(B.todense())

features = features[np.newaxis]
adj = adj[np.newaxis]
B = B[np.newaxis]
y_train = y_train[np.newaxis]
y_val = y_val[np.newaxis]
y_test = y_test[np.newaxis]
train_mask = train_mask[np.newaxis]
val_mask = val_mask[np.newaxis]
test_mask = test_mask[np.newaxis]

biases = process.adj_to_bias(adj, [nb_nodes], nhood=1)
biases_B = process.adj_to_bias(B, [nb_nodes], nhood=1)

# augmentation: edge, mask, angle, subgraph. positive samples
if aug_type == 'edge': # GraphCL
    aug_features = features[0]
    aug_adj = aug.aug_random_edge(adj[0], aug_percent)
elif aug_type == 'mask':
    aug_features = aug.aug_mask_ftr(features[0], aug_percent)
    aug_adj = adj[0]
elif aug_type == 'angle':
    aug_features, aug_adj, aug_B, aug_angle = aug.aug_drop_angle(features[0], adj[0], B[0], angle_vertex, aug_percent)
elif aug_type == 'subgraph':
    aug_features, aug_adj = aug.aug_subgraph(features[0], adj[0], aug_percent)
else:
    assert False

# shuffle: negative samples
idx = np.random.permutation(nb_nodes)
features_shuf = features[:, idx, :]

aug_features = aug_features[np.newaxis]
aug_adj = aug_adj[np.newaxis]
aug_B = aug_B[np.newaxis]
biases_aug = process.adj_to_bias(aug_adj, [nb_nodes], nhood=1)
biases_B_aug = process.adj_to_bias(aug_B, [nb_nodes], nhood=1)



with tf.Graph().as_default():
    with tf.name_scope('input'):
        ftr_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, nb_nodes, ft_size))
        ftr_shuf = tf.placeholder(dtype=tf.float32, shape=(batch_size, nb_nodes, ft_size))
        ftr_aug = tf.placeholder(dtype=tf.float32, shape=(batch_size, nb_nodes, ft_size))
        bias_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, nb_nodes, nb_nodes))
        bias_aug = tf.placeholder(dtype=tf.float32, shape=(batch_size, nb_nodes, nb_nodes))
        bias_B_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, nb_nodes, nb_nodes))
        bias_B_aug = tf.placeholder(dtype=tf.float32, shape=(batch_size, nb_nodes, nb_nodes))
        lbl_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes, nb_classes))
        msk_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes))
        attn_drop = tf.placeholder(dtype=tf.float32, shape=())
        ffd_drop = tf.placeholder(dtype=tf.float32, shape=())
        is_train = tf.placeholder(dtype=tf.bool, shape=())

    lbl_1 = tf.ones([batch_size, nb_nodes])
    lbl_2 = tf.zeros([batch_size, nb_nodes])
    lbl_pre = tf.concat([lbl_1, lbl_2], axis=1)

    logits_pre = dgi.DGI_forward(model, ftr_in, ftr_shuf, ftr_aug, bias_in, bias_B_in,
                    angle_vertex, bias_aug, bias_B_aug, aug_angle, nb_classes, nb_nodes,
                    attn_drop, ffd_drop, hid_units, n_heads)
    loss_pre = dgi.DGI_loss(logits_pre, lbl_pre)

    logits, _, affine_trans, harmonic_maps = model.inference(ftr_in, nb_classes,
                                nb_nodes, is_train, attn_drop, ffd_drop,
                                bias_mat=bias_in,
                                bias_mat_B=bias_B_in,
                                angle_vertex=angle_vertex, 
                                hid_units=hid_units, n_heads=n_heads,
                                residual=residual, activation=nonlinearity)
    log_resh = tf.reshape(logits, [-1, nb_classes])
    lab_resh = tf.reshape(lbl_in, [-1, nb_classes])
    msk_resh = tf.reshape(msk_in, [-1])
    loss_harm = process.harmonic_loss(K, harmonic_maps)
    loss = model.masked_softmax_cross_entropy(log_resh, lab_resh, msk_resh)
    accuracy = model.masked_accuracy(log_resh, lab_resh, msk_resh)

    train_op = model.training(loss + lam*loss_harm + mu*loss_pre, lr, l2_coef)

    saver = tf.train.Saver()

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    vlss_mn = np.inf
    vacc_mx = 0.0
    curr_step = 0

    with tf.Session() as sess:
        sess.run(init_op)

        train_loss_avg = 0
        train_acc_avg = 0
        train_loss_harm_avg = 0
        train_loss_pre_avg = 0
        val_loss_avg = 0
        val_acc_avg = 0
        val_loss_harm_avg = 0
        val_loss_pre_avg = 0

        for epoch in range(nb_epochs):
            tr_step = 0
            tr_size = features.shape[0]

            while tr_step * batch_size < tr_size:
                _, loss_value_tr, loss_harm_tr, loss_pre_tr, acc_tr = sess.run([train_op, loss, loss_harm, loss_pre, accuracy],
                    feed_dict={
                        ftr_in: features[tr_step*batch_size:(tr_step+1)*batch_size],
                        ftr_shuf: features_shuf[tr_step*batch_size:(tr_step+1)*batch_size],
                        ftr_aug: aug_features[tr_step*batch_size:(tr_step+1)*batch_size],
                        bias_in: biases[tr_step*batch_size:(tr_step+1)*batch_size],
                        bias_aug: biases_aug[tr_step*batch_size:(tr_step+1)*batch_size],
                        bias_B_in: biases_B[tr_step*batch_size:(tr_step+1)*batch_size],
                        bias_B_aug: biases_B_aug[tr_step*batch_size:(tr_step+1)*batch_size],
                        lbl_in: y_train[tr_step*batch_size:(tr_step+1)*batch_size],
                        msk_in: train_mask[tr_step*batch_size:(tr_step+1)*batch_size],
                        is_train: True,
                        attn_drop: 0.5, ffd_drop: 0.5})
                train_loss_avg += loss_value_tr
                train_acc_avg += acc_tr
                train_loss_harm_avg += loss_harm_tr
                train_loss_pre_avg += loss_pre_tr
                tr_step += 1

            vl_step = 0
            vl_size = features.shape[0]

            while vl_step * batch_size < vl_size:
                loss_value_vl, loss_harm_vl, loss_pre_vl, acc_vl = sess.run([loss, loss_harm, loss_pre, accuracy],
                    feed_dict={
                        ftr_in: features[vl_step*batch_size:(vl_step+1)*batch_size],
                        ftr_shuf: features_shuf[vl_step*batch_size:(vl_step+1)*batch_size],
                        ftr_aug: aug_features[vl_step*batch_size:(vl_step+1)*batch_size],
                        bias_in: biases[vl_step*batch_size:(vl_step+1)*batch_size],
                        bias_aug: biases_aug[vl_step*batch_size:(vl_step+1)*batch_size],
                        bias_B_in: biases_B[vl_step*batch_size:(vl_step+1)*batch_size],
                        bias_B_aug: biases_B_aug[vl_step*batch_size:(vl_step+1)*batch_size],
                        lbl_in: y_val[vl_step*batch_size:(vl_step+1)*batch_size],
                        msk_in: val_mask[vl_step*batch_size:(vl_step+1)*batch_size],
                        is_train: False,
                        attn_drop: 0.0, ffd_drop: 0.0})
                val_loss_avg += loss_value_vl
                val_acc_avg += acc_vl
                val_loss_harm_avg += loss_harm_vl
                val_loss_pre_avg += loss_pre_vl
                vl_step += 1

            print('Training: loss = %.5f, acc = %.5f, harm = %.5f, pre = %.5f | Val: loss = %.5f, acc = %.5f, harm = %.5f, pre = %.5f' %
                    (train_loss_avg/tr_step, train_acc_avg/tr_step, train_loss_harm_avg/tr_step, train_loss_pre_avg/tr_step,
                    val_loss_avg/vl_step, val_acc_avg/vl_step, val_loss_harm_avg/vl_step, val_loss_pre_avg/vl_step))

            # early stop
            if val_acc_avg/vl_step >= vacc_mx or val_loss_avg/vl_step <= vlss_mn:
                if val_acc_avg/vl_step >= vacc_mx and val_loss_avg/vl_step <= vlss_mn:
                    vacc_early_model = val_acc_avg/vl_step
                    vlss_early_model = val_loss_avg/vl_step
                    saver.save(sess, checkpt_file)
                vacc_mx = np.max((val_acc_avg/vl_step, vacc_mx)) # max accuracy on validation set
                vlss_mn = np.min((val_loss_avg/vl_step, vlss_mn)) # min loss on validation set
                curr_step = 0 # note that once the model ability improves, "curr_step" will be cleared and restart counting
            else:
                curr_step += 1 # model ability didn't improve in successive rounds
                if curr_step == patience:
                    print('Early stop! Min loss: ', vlss_mn, ', Max accuracy: ', vacc_mx)
                    print('Early stop model validation loss: ', vlss_early_model, ', accuracy: ', vacc_early_model)
                    break

            train_loss_avg = 0
            train_acc_avg = 0
            train_loss_harm_avg = 0
            train_loss_pre_avg = 0
            val_loss_avg = 0
            val_acc_avg = 0
            val_loss_harm_avg = 0
            val_loss_pre_avg = 0

        saver.restore(sess, checkpt_file)

        ts_step = 0
        ts_size = features.shape[0]
        ts_loss = 0.0
        ts_acc = 0.0

        while ts_step * batch_size < ts_size:
            loss_value_ts, acc_ts, affine, harmonic = sess.run([loss, accuracy, affine_trans, harmonic_maps],
                feed_dict={
                    ftr_in: features[ts_step*batch_size:(ts_step+1)*batch_size],
                    ftr_shuf: features_shuf[ts_step*batch_size:(ts_step+1)*batch_size],
                    ftr_aug: aug_features[ts_step*batch_size:(ts_step+1)*batch_size],
                    bias_in: biases[ts_step*batch_size:(ts_step+1)*batch_size],
                    bias_aug: biases_aug[ts_step*batch_size:(ts_step+1)*batch_size],
                    bias_B_in: biases_B[ts_step*batch_size:(ts_step+1)*batch_size],
                    bias_B_aug: biases_B_aug[ts_step*batch_size:(ts_step+1)*batch_size],
                    lbl_in: y_test[ts_step*batch_size:(ts_step+1)*batch_size],
                    msk_in: test_mask[ts_step*batch_size:(ts_step+1)*batch_size],
                    is_train: False,
                    attn_drop: 0.0, ffd_drop: 0.0})
            ts_loss += loss_value_ts
            ts_acc += acc_ts
            ts_step += 1

        with open("map_result.pickle", 'wb') as f:
            pkl.dump([affine, harmonic], f)

        print('Test loss:', ts_loss/ts_step, '; Test accuracy:', ts_acc/ts_step)

        sess.close()
