import tensorflow as tf


def bilinear(input1, input2):
    matmul = tf.layers.dense(input1, input2.shape[-1], kernel_initializer=tf.glorot_uniform_initializer(),
                             use_bias=False, activation=None)
    elemul = matmul * input2
    output = tf.reduce_sum(elemul, axis=-1)
    return output


def discriminator(c, h_pl, h_mi):
    c_x = tf.expand_dims(c, 1)
    c_x = tf.tile(c_x, multiples=[1,h_pl.shape[1],1])
    sc_1 = bilinear(h_pl, c_x)
    sc_2 = bilinear(h_mi, c_x)
    logits = tf.concat([sc_1, sc_2], axis=1)
    return logits


def DGI_forward(model, seq, seq_shuf, seq_aug, adj, adj_B, angle, adj_aug, adj_B_aug, 
        angle_aug, nb_classes, nb_nodes, attn_drop, ffd_drop, hid_units, n_heads):
    h_0 = model.inference(seq, nb_classes, nb_nodes, training=True,
            attn_drop=attn_drop, ffd_drop=ffd_drop, bias_mat=adj, bias_mat_B=adj_B,
            angle_vertex=angle, hid_units=hid_units, n_heads=n_heads,
            activation=tf.nn.elu, residual=False)[1]

    h_2 = model.inference(seq_shuf, nb_classes, nb_nodes, training=True,
            attn_drop=attn_drop, ffd_drop=ffd_drop, bias_mat=adj, bias_mat_B=adj_B,
            angle_vertex=angle, hid_units=hid_units, n_heads=n_heads,
            activation=tf.nn.elu, residual=False)[1]

    h_1 = model.inference(seq_aug, nb_classes, nb_nodes, training=True,
            attn_drop=attn_drop, ffd_drop=ffd_drop, bias_mat=adj_aug, bias_mat_B=adj_B_aug,
            angle_vertex=angle_aug, hid_units=hid_units, n_heads=n_heads,
            activation=tf.nn.elu, residual=False)[1]

    c = tf.reduce_mean(h_1, axis=1)
    c = tf.nn.sigmoid(c)

    ret = discriminator(c, h_0, h_2)

    return ret


def DGI_loss(logits, labels):
    b_xent = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    return tf.reduce_mean(b_xent)


def DGI_training(loss, lr=0.001, l2_coef=0.0):
    # weight decay
    vars = tf.trainable_variables()
    lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars if v.name not
                       in ['bias', 'gamma', 'b', 'g', 'beta']]) * l2_coef

    # optimizer
    opt = tf.train.AdamOptimizer(learning_rate=lr)

    # training op
    train_op = opt.minimize(loss + lossL2)

    return train_op
