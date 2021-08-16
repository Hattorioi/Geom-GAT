import numpy as np
import tensorflow as tf

from utils import layers
from models.base_gattn import BaseGAttN

class SpGeomGAT(BaseGAttN):
    def inference(inputs, nb_classes, nb_nodes, training, attn_drop, ffd_drop,
            bias_mat, bias_mat_B, angle_vertex, hid_units, n_heads, activation=tf.nn.elu, residual=False):
        attns = []
        affine_trans = []
        harmonic_maps = []
        for _ in range(n_heads[0]):
            x = layers.sp_attn_head(inputs, adj_mat=bias_mat,
                    out_sz=hid_units[0], activation=activation, nb_nodes=nb_nodes,
                    in_drop=ffd_drop, coef_drop=attn_drop, residual=False)
            attns.append(x[0])
            affine_trans.append(x[1])
            y = layers.sp_attn_head_angle(inputs, adj_mat=bias_mat_B, angle_vertex=angle_vertex,
                    out_sz=hid_units[0], activation=activation, nb_nodes=nb_nodes,
                    in_drop=ffd_drop, coef_drop=attn_drop, residual=False)
            attns.append(y[0])
            harmonic_maps.append(y[1])
        h_1 = tf.concat(attns, axis=-1)
        out = []
        for i in range(n_heads[-1]):
            x = layers.sp_attn_head(h_1, adj_mat=bias_mat,
                    out_sz=nb_classes, activation=lambda x: x, nb_nodes=nb_nodes,
                    in_drop=ffd_drop, coef_drop=attn_drop, residual=False)
            out.append(x[0])
        logits = tf.add_n(out) / n_heads[-1]
    
        return logits, h_1, affine_trans, harmonic_maps
