import time
import numpy as np
import scipy.sparse as sp
import tensorflow as tf

from models import SpGeomGAT
from utils import process

checkpt_file = 'pre_trained/cora/mod_cora.ckpt'

dataset = 'pubmed'

# training params
batch_size = 1
nb_epochs = 100000
patience = 100
lr = 0.005  # learning rate
l2_coef = 0.001  # weight decay
hid_units = [8] # numbers of hidden units per each attention head in each layer
n_heads = [4, 2] # additional entry for the output layer
residual = False
nonlinearity = tf.nn.elu
lam = 0.01 # coefficient of harmonic energy
model = SpGeomGAT

print('Dataset: ' + dataset)
print('----- Opt. hyperparams -----')
print('lr: ' + str(lr))
print('l2_coef: ' + str(l2_coef))
print('lambda: ' + str(lam))
print('----- Archi. hyperparams -----')
print('nb. layers: ' + str(len(hid_units)))
print('nb. units per layer: ' + str(hid_units))
print('nb. attention heads: ' + str(n_heads))
print('residual: ' + str(residual))
print('nonlinearity: ' + str(nonlinearity))
print('model: ' + str(model))

sparse = True

adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = process.load_data(dataset)
features, spars = process.preprocess_features(features)
B, angle_vertex = process.compute_vertex(adj, features)
K = process.sp_spring_constants(angle_vertex, features)

nb_nodes = features.shape[0]
ft_size = features.shape[1]
nb_classes = y_train.shape[1]

features = features[np.newaxis]
y_train = y_train[np.newaxis]
y_val = y_val[np.newaxis]
y_test = y_test[np.newaxis]
train_mask = train_mask[np.newaxis]
val_mask = val_mask[np.newaxis]
test_mask = test_mask[np.newaxis]

biases = process.preprocess_adj_bias(adj)
biases_B = process.preprocess_adj_bias(B)

with tf.Graph().as_default():
    with tf.name_scope('input'):
        ftr_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, nb_nodes, ft_size))
        if sparse:
            bias_in = tf.sparse_placeholder(dtype=tf.float32)
            bias_B_in = tf.sparse_placeholder(dtype=tf.float32)
        else:
            bias_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, nb_nodes, nb_nodes))
            bias_B_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, nb_nodes, nb_nodes))
        lbl_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes, nb_classes))
        msk_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes))
        attn_drop = tf.placeholder(dtype=tf.float32, shape=())
        ffd_drop = tf.placeholder(dtype=tf.float32, shape=())
        is_train = tf.placeholder(dtype=tf.bool, shape=())

    logits, _, affine_trans, harmonic_maps = model.inference(ftr_in, nb_classes, nb_nodes, is_train,
                                attn_drop, ffd_drop,
                                bias_mat=bias_in,
                                bias_mat_B=bias_B_in,
                                angle_vertex=angle_vertex,
                                hid_units=hid_units, n_heads=n_heads,
                                residual=residual, activation=nonlinearity)
    log_resh = tf.reshape(logits, [-1, nb_classes])
    lab_resh = tf.reshape(lbl_in, [-1, nb_classes])
    msk_resh = tf.reshape(msk_in, [-1])
    loss_harm = process.sp_harmonic_loss(K, harmonic_maps)
    loss = model.masked_softmax_cross_entropy(log_resh, lab_resh, msk_resh)
    accuracy = model.masked_accuracy(log_resh, lab_resh, msk_resh)

    train_op = model.training(loss + lam*loss_harm, lr, l2_coef)

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
        val_loss_avg = 0
        val_acc_avg = 0
        val_loss_harm_avg = 0

        for epoch in range(nb_epochs):
            tr_step = 0
            tr_size = features.shape[0]

            while tr_step * batch_size < tr_size:
                if sparse:
                    bbias = biases
                    bbias_B = biases_B
                else:
                    bbias = biases[tr_step*batch_size:(tr_step+1)*batch_size]
                    bbias_B = biases_B[tr_step*batch_size:(tr_step+1)*batch_size]

                _, loss_value_tr, loss_harm_tr, acc_tr = sess.run([train_op, loss, loss_harm, accuracy],
                    feed_dict={
                        ftr_in: features[tr_step*batch_size:(tr_step+1)*batch_size],
                        bias_in: bbias,
                        bias_B_in: bbias_B,
                        lbl_in: y_train[tr_step*batch_size:(tr_step+1)*batch_size],
                        msk_in: train_mask[tr_step*batch_size:(tr_step+1)*batch_size],
                        is_train: True,
                        attn_drop: 0.5, ffd_drop: 0.5})
                train_loss_avg += loss_value_tr
                train_acc_avg += acc_tr
                train_loss_harm_avg += loss_harm_tr
                tr_step += 1

            vl_step = 0
            vl_size = features.shape[0]

            while vl_step * batch_size < vl_size:
                if sparse:
                    bbias = biases
                    bbias_B = biases_B
                else:
                    bbias = biases[vl_step*batch_size:(vl_step+1)*batch_size]
                    bbias_B = biases_B[vl_step*batch_size:(vl_step+1)*batch_size]
                loss_value_vl, loss_harm_vl, acc_vl = sess.run([loss, loss_harm, accuracy],
                    feed_dict={
                        ftr_in: features[vl_step*batch_size:(vl_step+1)*batch_size],
                        bias_in: bbias,
                        bias_B_in: bbias_B,
                        lbl_in: y_val[vl_step*batch_size:(vl_step+1)*batch_size],
                        msk_in: val_mask[vl_step*batch_size:(vl_step+1)*batch_size],
                        is_train: False,
                        attn_drop: 0.0, ffd_drop: 0.0})
                val_loss_avg += loss_value_vl
                val_acc_avg += acc_vl
                val_loss_harm_avg += loss_harm_vl
                vl_step += 1

            print('Training: loss = %.5f, acc = %.5f, harm = %.5f | Val: loss = %.5f, acc = %.5f, harm = %.5f' %
                    (train_loss_avg/tr_step, train_acc_avg/tr_step, train_loss_harm_avg/tr_step,
                    val_loss_avg/vl_step, val_acc_avg/vl_step, val_loss_harm_avg/vl_step))

            if val_acc_avg/vl_step >= vacc_mx or val_loss_avg/vl_step <= vlss_mn:
                if val_acc_avg/vl_step >= vacc_mx and val_loss_avg/vl_step <= vlss_mn:
                    vacc_early_model = val_acc_avg/vl_step
                    vlss_early_model = val_loss_avg/vl_step
                    saver.save(sess, checkpt_file)
                vacc_mx = np.max((val_acc_avg/vl_step, vacc_mx))
                vlss_mn = np.min((val_loss_avg/vl_step, vlss_mn))
                curr_step = 0
            else:
                curr_step += 1
                if curr_step == patience:
                    print('Early stop! Min loss: ', vlss_mn, ', Max accuracy: ', vacc_mx)
                    print('Early stop model validation loss: ', vlss_early_model, ', accuracy: ', vacc_early_model)
                    break

            train_loss_avg = 0
            train_acc_avg = 0
            train_loss_harm_avg = 0
            val_loss_avg = 0
            val_acc_avg = 0
            val_loss_harm_avg = 0

        saver.restore(sess, checkpt_file)

        ts_step = 0
        ts_size = features.shape[0]
        ts_loss = 0.0
        ts_acc = 0.0

        while ts_step * batch_size < ts_size:
            if sparse:
                bbias = biases
                bbias_B = biases_B
            else:
                bbias = biases[ts_step*batch_size:(ts_step+1)*batch_size]
                bbias_B = biases_B[ts_step*batch_size:(ts_step+1)*batch_size]
            loss_value_ts, acc_ts = sess.run([loss, accuracy],
                feed_dict={
                    ftr_in: features[ts_step*batch_size:(ts_step+1)*batch_size],
                    bias_in: bbias,
                    bias_B_in: bbias_B,
                    lbl_in: y_test[ts_step*batch_size:(ts_step+1)*batch_size],
                    msk_in: test_mask[ts_step*batch_size:(ts_step+1)*batch_size],
                    is_train: False,
                    attn_drop: 0.0, ffd_drop: 0.0})
            ts_loss += loss_value_ts
            ts_acc += acc_ts
            ts_step += 1

        print('Test loss:', ts_loss/ts_step, '; Test accuracy:', ts_acc/ts_step)

        sess.close()
