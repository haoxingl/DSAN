from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential
from tensorflow.keras.utils import get_custom_objects


def gelu(x):
    return 0.5 * x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))


get_custom_objects().update({'gelu': layers.Activation(gelu)})

act = 'relu'


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def spatial_posenc(position_r, position_c, d_model):
    angle_rads_r = get_angles(position_r, np.arange(d_model)[np.newaxis, :], d_model)

    angle_rads_c = get_angles(position_c, np.arange(d_model)[np.newaxis, :], d_model)

    pos_encoding = np.zeros(angle_rads_r.shape, dtype=np.float32)

    pos_encoding[:, 0::2] = np.sin(angle_rads_r[:, 0::2])

    pos_encoding[:, 1::2] = np.cos(angle_rads_c[:, 1::2])

    return tf.cast(pos_encoding[np.newaxis, ...], dtype=tf.float32)


class Convs(layers.Layer):
    def __init__(self, n_layer, n_filter, l_hist, r_d=0.1):
        super(Convs, self).__init__()

        self.n_layer = n_layer
        self.l_hist = l_hist

        self.convs = [[layers.Conv2D(n_filter, (3, 3), activation=act, padding='same')
                       for _ in range(l_hist)] for _ in range(n_layer)]
        # self.batchnorms = [[layers.BatchNormalization(epsilon=1e-6) for _ in range(l_hist)] for _ in range(n_layer)]
        self.dropouts = [[layers.Dropout(r_d) for _ in range(l_hist)] for _ in range(n_layer)]

    def call(self, inps, training):
        outputs = tf.split(inps, self.l_hist, axis=1)
        for i in range(self.n_layer):
            for j in range(self.l_hist):
                if i == 0:
                    outputs[j] = tf.squeeze(outputs[j], axis=1)
                outputs[j] = self.convs[i][j](outputs[j])
                outputs[j] = self.dropouts[i][j](outputs[j], training=training)
                if i == self.n_layer - 1:
                    outputs[j] = tf.expand_dims(outputs[j], axis=1)

        output = tf.concat(outputs, axis=1)

        return output


def scaled_dot_product_attention(q, k, v, mask=None):
    matmul_qk = tf.matmul(q, k, transpose_b=True)

    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    output = tf.matmul(attention_weights, v)

    return output, attention_weights


class MultiHeadAttention(layers.Layer):
    def __init__(self, d_model, n_head, self_att=True):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.self_att = self_att

        assert d_model % n_head == 0

        self.depth = d_model // n_head

        if self_att:
            self.wx = layers.Dense(d_model * 3)
        else:
            self.wq = layers.Dense(d_model)
            self.wkv = layers.Dense(d_model * 2)

        self.wo = layers.Dense(d_model)

    def split_heads(self, x):
        shape = tf.shape(x)
        x = tf.reshape(x, (shape[0], shape[1], shape[2], self.n_head, self.depth))
        return tf.transpose(x, perm=[0, 1, 3, 2, 4])

    def call(self, v, k, q, mask):
        if self.self_att:
            q, k, v = tf.split(self.wx(q), 3, axis=-1)
        else:
            q = self.wq(q)
            k, v = tf.split(self.wkv(k), 2, axis=-1)

        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 1, 3, 2, 4])

        d_shape = tf.shape(scaled_attention)

        concat_attention = tf.reshape(scaled_attention, (d_shape[0], d_shape[1], d_shape[2], self.d_model))

        output = self.wo(concat_attention)

        return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    return Sequential([
        layers.Dense(dff, activation=act),
        layers.Dense(d_model)
    ])


def ex_encoding(d_model, dff):
    return Sequential([
        layers.Dense(dff),
        layers.Dense(d_model, activation='sigmoid')
    ])


class EncoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, dff, dpo_rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(dpo_rate)
        self.dropout2 = layers.Dropout(dpo_rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2


class DecoderLayer(layers.Layer):
    def __init__(self, d_model, n_head, dff, r_d=0.1, revert_q=False):
        super(DecoderLayer, self).__init__()

        self.revert_q = revert_q

        self.mha1 = MultiHeadAttention(d_model, n_head)
        self.mha2 = MultiHeadAttention(d_model, n_head, self_att=False)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(r_d)
        self.dropout2 = layers.Dropout(r_d)
        self.dropout3 = layers.Dropout(r_d)

    def call(self, x, kv, training, look_ahead_mask, padding_mask):
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        if self.revert_q:
            out1 = tf.transpose(out1, perm=[0, 2, 1, 3])

        attn2, attn_weights_block2 = self.mha2(kv, kv, out1, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)

        if self.revert_q:
            out2 = tf.transpose(out2, perm=[0, 2, 1, 3])

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)

        return out3, attn_weights_block1, attn_weights_block2


class DAE(layers.Layer):
    def __init__(self, n_layer, d_model, n_head, dff, conv_layer, conv_filter, l_hist, r_d=0.1):
        super(DAE, self).__init__()

        self.d_model = d_model
        self.n_layer = n_layer

        self.convs = Convs(conv_layer, conv_filter, l_hist, r_d)
        self.convs_g = Convs(conv_layer, conv_filter, l_hist, r_d)
        # self.convs = layers.Dense(d_model, activation=act)
        # self.convs_g = layers.Dense(d_model, activation=act)

        self.ex_encoder = ex_encoding(d_model, dff)
        self.ex_encoder_g = ex_encoding(d_model, dff)

        self.dropout = layers.Dropout(r_d)
        self.dropout_g = layers.Dropout(r_d)

        self.enc_g = [EncoderLayer(d_model, n_head, dff, r_d) for _ in range(n_layer)]
        self.enc_l = [DecoderLayer(d_model, n_head, dff, r_d) for _ in range(n_layer)]

    def call(self, x, x_g, ex, cors, cors_g, training, padding_mask, padding_mask_g):
        attention_weights = {}

        shape = tf.shape(x)

        ex_enc = tf.expand_dims(self.ex_encoder(ex), axis=2)
        ex_enc_g = tf.expand_dims(self.ex_encoder_g(ex), axis=2)
        pos_enc = tf.expand_dims(cors, axis=1)
        pos_enc_g = tf.expand_dims(cors_g, axis=1)

        x = self.convs(x, training)
        x_g = self.convs_g(x_g, training)
        # x = self.convs(x)
        # x_g = self.convs_g(x_g)

        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x_g *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        x = tf.reshape(x, [shape[0], shape[1], -1, self.d_model])
        x_g = tf.reshape(x_g, [shape[0], shape[1], -1, self.d_model])

        x = x + ex_enc + pos_enc
        x_g = x_g + ex_enc_g + pos_enc_g

        x = self.dropout(x, training=training)
        x_g = self.dropout_g(x_g, training=training)

        for i in range(self.n_layer):
            x_g = self.enc_g[i](x_g, training, padding_mask_g)

        for i in range(self.n_layer):
            x, block1, block2 = self.enc_l[i](x, x_g, training, padding_mask, padding_mask_g)
            attention_weights['dae_layer{}_block1'.format(i + 1)] = block1
            attention_weights['dae_layer{}_block2'.format(i + 1)] = block2

        return x, attention_weights


class SAD(layers.Layer):
    def __init__(self, n_layer, d_model, n_head, dff, conv_layer, r_d=0.1):
        super(SAD, self).__init__()

        self.d_model = d_model
        self.n_layer = n_layer
        self.pos_enc = spatial_posenc(0, 0, self.d_model)

        self.ex_encoder = ex_encoding(d_model, dff)
        self.dropout = layers.Dropout(r_d)

        self.li_conv = Sequential([layers.Dense(d_model, activation=act) for _ in range(conv_layer)])
        # self.li_conv = layers.Dense(d_model, activation=act)

        self.dec_s = [DecoderLayer(d_model, n_head, dff, r_d) for _ in range(n_layer)]
        self.dec_t = [DecoderLayer(d_model, n_head, dff, r_d, revert_q=True) for _ in range(n_layer)]

    def call(self, x, ex, dae_output, training, look_ahead_mask, padding_mask):
        attention_weights = {}

        ex_enc = self.ex_encoder(ex)

        x = self.li_conv(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + ex_enc + self.pos_enc

        x = self.dropout(x, training=training)
        x_s = tf.expand_dims(x, axis=1)
        x_t = tf.expand_dims(x, axis=1)

        for i in range(self.n_layer):
            x_s, block1, block2 = self.dec_s[i](x_s, dae_output, training, look_ahead_mask, None)
            attention_weights['sad_s_layer{}_block1'.format(i + 1)] = block1
            attention_weights['sad_s_layer{}_block2'.format(i + 1)] = block2

        x_s = tf.transpose(x_s, perm=[0, 2, 1, 3])

        for i in range(self.n_layer):
            x_t, block1, block2 = self.dec_t[i](x_t, x_s, training, look_ahead_mask, None)
            attention_weights['decoder_t_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_t_layer{}_block2'.format(i + 1)] = block2

        output = tf.squeeze(x_t, axis=1)

        return output, attention_weights


class DSAN(Model):
    def __init__(self, n_layer, d_model, n_head, dff, conv_layer, conv_filter, l_hist, r_d=0.1):
        super(DSAN, self).__init__()

        self.dae = DAE(n_layer, d_model, n_head, dff, conv_layer, conv_filter, l_hist, r_d)

        self.sad = SAD(n_layer, d_model, n_head, dff, conv_layer, r_d)

        self.final_layer = layers.Dense(2, activation='tanh')

    def call(self, dae_inp_g, dae_inp, dae_inp_ex, sad_inp, sad_inp_ex, cors, cors_g, training,
             padding_mask, padding_mask_g, look_ahead_mask):
        dae_output, attention_weights_dae = \
            self.dae(dae_inp, dae_inp_g, dae_inp_ex, cors, cors_g, training, padding_mask, padding_mask_g)

        sad_output, attention_weights_sad = \
            self.sad(sad_inp, sad_inp_ex, dae_output, training, look_ahead_mask, None)

        final_output = self.final_layer(sad_output)

        return final_output, attention_weights_dae, attention_weights_sad
