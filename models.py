from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model, Sequential
from tensorflow.keras.activations import sigmoid

actfunc = 'relu'
# from tensorflow_addons.activations import gelu as actfunc
# from tensorflow_addons import layers as tfa_layers

final_cnn_filters = 128


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


def spatial_posenc_batch(position_r, position_c, d_model):
    angle_rads_r = get_angles(position_r[..., np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)

    angle_rads_c = get_angles(position_c[..., np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)

    pos_encoding = np.zeros(angle_rads_r.shape, dtype=np.float32)

    pos_encoding[..., 0::2] = np.sin(angle_rads_r[..., 0::2])

    pos_encoding[..., 1::2] = np.cos(angle_rads_c[..., 1::2])

    return tf.cast(pos_encoding, dtype=tf.float32)


# """ the local convolutional layer before the encoder and deconder stacks """
# class Local_Conv(layers.Layer):
#     def __init__(self, num_layers, num_filters, num_intervals, dpo_rate=0.1):
#         super(Local_Conv, self).__init__()
#
#         self.num_intervals = num_intervals  # indicate how many time intervals are included in the historical inputs
#         self.num_layers = num_layers
#
#         """ data from each time interval will be handled by one set of convolutional layers, therefore totally
#             totally num_intervals sets of convolutional layers are employed """
#         self.conv_layers = [[layers.Conv2D(num_filters, (3, 3), activation=actfunc, padding='same')
#                              for _ in range(num_layers)] for _ in range(num_intervals)]
#
#         self.dropout_layers = [[layers.Dropout(dpo_rate) for _ in range(num_layers)] for _ in range(num_intervals)]
#
#     def call(self, inputs, training):
#         outputs = []
#         for i in range(self.num_intervals):
#             output = inputs[:, :, :, i, :]
#             for j in range(self.num_layers):
#                 output = self.conv_layers[i][j](output)
#                 output = self.dropout_layers[i][j](output, training=training)
#             # output = self.dropout_layers[i](output, training=training)
#             output = tf.expand_dims(output, axis=3)
#             outputs.append(output)
#
#         output_final = tf.concat(outputs, axis=3)
#
#         return output_final


""" the implementation of the Gated Fusion mechanism, check the details in my paper """


class Gated_Conv(layers.Layer):
    def __init__(self, num_layers, num_filters, seq_len, dpo_rate=0.1):
        super(Gated_Conv, self).__init__()

        self.seq_len = seq_len  # indicate how many time intervals are included in the historical inputs
        self.num_layers = num_layers

        self.conv_layers_f = [[layers.Conv2D(num_filters, (3, 3), activation=actfunc, padding='same')
                               for _ in range(num_layers)] for _ in range(seq_len)]
        self.conv_layers_t = [[layers.Conv2D(num_filters, (3, 3), activation=actfunc, padding='same')
                               for _ in range(num_layers)] for _ in range(seq_len)]

        self.sigm = [[layers.Activation(sigmoid) for _ in range(num_layers)] for _ in range(seq_len)]

        self.dpo_layers = [layers.Dropout(dpo_rate * num_layers) for _ in range(seq_len)]

    def call(self, inputs_flow, inputs_trans, training):
        outputs_f = []

        for i in range(self.seq_len):
            output_f = inputs_flow[:, :, :, i, :]
            output_t = inputs_trans[:, :, :, i, :]
            for j in range(self.num_layers):
                output_t = self.conv_layers_t[i][j](output_t)
                output_f = self.conv_layers_f[i][j](output_f) * self.sigm[i][j](output_t)
            output_f = self.dpo_layers[i](output_f, training=training)
            output_f = tf.expand_dims(output_f, axis=3)
            outputs_f.append(output_f)

        output_final = tf.concat(outputs_f, axis=3)

        return output_final


# class Gated_Conv_1(layers.Layer):
#     def __init__(self, num_layers, num_filters, num_intervals, dpo_rate=0.1):
#         super(Gated_Conv_1, self).__init__()
#
#         self.num_intervals = num_intervals  # indicate how many time intervals are included in the historical inputs
#         self.num_layers = num_layers
#
#         self.conv_layers_flow = [[layers.Conv2D(num_filters, (3, 3), activation=actfunc, padding='same')
#                                   for _ in range(num_layers)] for _ in range(num_intervals)]
#         self.conv_layers_trans = [[layers.Conv2D(num_filters, (3, 3), activation=actfunc, padding='same')
#                                    for _ in range(num_layers)] for _ in range(num_intervals)]
#
#         self.sigm = [[layers.Activation(sigmoid) for _ in range(num_layers)] for _ in range(num_intervals)]
#
#         self.dropout_layers = [layers.Dropout(dpo_rate) for _ in range(num_intervals)]
#
#     def call(self, inputs_flow, inputs_trans, training):
#         outputs_f = []
#
#         for i in range(self.num_intervals):
#             output_f = inputs_flow[:, :, :, i, :]
#             output_t = inputs_trans[:, :, :, i, :]
#             for j in range(self.num_layers):
#                 output_t = self.conv_layers_trans[i][j](output_t)
#                 output_f = self.conv_layers_flow[i][j](output_f) * self.sigm[i][j](output_t)
#             output_f = self.dropout_layers[i](output_f, training=training)
#             output_f = tf.expand_dims(output_f, axis=3)
#             outputs_f.append(output_f)
#
#         output_final = tf.concat(outputs_f, axis=3)
#
#         return output_final
#
#
# class Gated_Conv_2(layers.Layer):
#     def __init__(self, d_final, dpo_rate=0.1):
#         super(Gated_Conv_2, self).__init__()
#
#         self.conv_layers_flow = [layers.Conv2D(final_cnn_filters, (3, 3), activation=actfunc) for i in range(2)]
#         self.conv_layers_trans = [layers.Conv2D(final_cnn_filters, (3, 3), activation=actfunc) for i in range(2)]
#
#         self.dense1 = layers.Dense(d_final, activation=actfunc)
#         self.dense2 = layers.Dense(2, activation='tanh')
#
#         self.flatten = layers.Flatten()
#         self.dropout = layers.Dropout(dpo_rate)
#
#     def call(self, input_flow, input_trans, training):
#         input_trans = tf.squeeze(input_trans, axis=-2)
#         input_flow = tf.squeeze(input_flow, axis=-2) * self.sigm[0](input_trans)
#
#         trans_output1 = self.conv_layers_trans[0](input_trans)
#         flow_output1 = self.conv_layers_flow[0](input_flow) * self.sigm[1](trans_output1)
#
#         trans_output2 = self.conv_layers_trans[1](trans_output1)
#         flow_output2 = self.conv_layers_flow[1](flow_output1) * self.sigm[2](trans_output2)
#
#         output = self.flatten(flow_output2)
#         output = self.dense1(output)
#         output = self.dropout(output, training=training)
#         output = self.dense2(output)
#
#         return output


def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)

    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    output = tf.matmul(attention_weights, v)

    return output, attention_weights


class MultiHeadAttention(layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)

        self.dense = layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        output = self.dense(concat_attention)

        return output, attention_weights


# """ the implementation of Spatial-Temporal Multi-Head Attention, detailed in my paper"""
#
#
# class SpatialTemporal_MultiHeadAttention(layers.Layer):
#     def __init__(self, d_model, num_heads):
#         super(SpatialTemporal_MultiHeadAttention, self).__init__()
#         self.num_heads = num_heads
#         self.d_model = d_model
#
#         assert d_model % self.num_heads == 0
#
#         self.depth = d_model // self.num_heads
#
#         self.wq = layers.Dense(d_model)
#         self.wk = layers.Dense(d_model)
#         self.wv = layers.Dense(d_model)
#
#         self.dense = layers.Dense(d_model)
#
#     def split_heads(self, x, batch_size):
#         x = tf.reshape(x, (batch_size, tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3], self.num_heads, self.depth))
#         return tf.transpose(x, perm=[0, 1, 2, 4, 3, 5])
#
#     def call(self, v, k, q, mask):
#         batch_size = tf.shape(q)[0]
#
#         q = self.wq(q)
#         k = self.wk(k)
#         v = self.wv(v)
#
#         q = self.split_heads(q, batch_size)
#         k = self.split_heads(k, batch_size)
#         v = self.split_heads(v, batch_size)
#
#         scaled_attention, attention_weights = scaled_dot_product_attention(
#             q, k, v, mask)
#
#         scaled_attention = tf.transpose(scaled_attention,
#                                         perm=[0, 1, 2, 4, 3, 5])
#
#         concat_attention = tf.reshape(scaled_attention,
#                                       (batch_size, tf.shape(scaled_attention)[1], tf.shape(scaled_attention)[2],
#                                        tf.shape(scaled_attention)[3], self.d_model))
#
#         output = self.dense(concat_attention)
#
#         return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    return Sequential([
        layers.Dense(dff, activation=actfunc),
        layers.Dense(d_model)
    ])


def ex_encoding(d_model):
    return Sequential([
        layers.Dense(d_model * 2, activation=actfunc),
        layers.Dense(d_model, activation='sigmoid')
    ])


class EncoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, dff=256, dpo_rate=0.1):
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
    def __init__(self, d_model, num_heads, dff=256, dpo_rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(dpo_rate)
        self.dropout2 = layers.Dropout(dpo_rate)
        self.dropout3 = layers.Dropout(dpo_rate)

    def call(self, x, enc_output_x, training, look_ahead_mask, padding_mask):
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(enc_output_x, enc_output_x, out1, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(out2 + ffn_output)

        return out3, attn_weights_block1, attn_weights_block2


class Encoder(layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, cnn_layers, cnn_filters, seq_len, dpo_rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.seq_len = seq_len

        self.ex_enc = ex_encoding(d_model)
        self.dropout = layers.Dropout(dpo_rate)

        self.gated_conv = Gated_Conv(cnn_layers, cnn_filters, seq_len, dpo_rate)
        self.gated_conv_t = Gated_Conv(cnn_layers, cnn_filters, seq_len, dpo_rate)

        self.sigm = layers.Activation(sigmoid)

        self.encs = [EncoderLayer(d_model, num_heads, dff, dpo_rate) for _ in range(num_layers)]

    def call(self, x, ex, cors, t_gate, training, mask=None):
        ex_enc = tf.expand_dims(self.dropout(self.ex_encoding(ex), training=training), axis=2)
        pos_enc = tf.expand_dims(spatial_posenc_batch(cors[..., 0], cors[..., 1], self.d_model))

        t_gate = self.sigm(self.gated_conv_t(t_gate, training))
        x = self.gated_conv(x, training) * t_gate
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x_shape = tf.shape(x)
        x = tf.reshape(x, [x_shape[0], x_shape[1], -1, x_shape[4]])
        x = x + ex_enc + pos_enc

        x = tf.reshape(x, [x_shape[0], -1, x_shape[4]])

        for i in range(self.num_layers):
            x = self.encs[i](x, training, mask)

        return x


class EncoderBR(layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, cnn_layers, cnn_filters, seq_len, dpo_rate=0.1):
        super(EncoderBR, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.seq_len = seq_len

        self.ex_enc = ex_encoding(d_model)
        self.dropout = layers.Dropout(dpo_rate)

        self.gated_conv = Gated_Conv(cnn_layers, cnn_filters, seq_len, dpo_rate)
        self.gated_conv_t = Gated_Conv(cnn_layers, cnn_filters, seq_len, dpo_rate)

        self.sigm = layers.Activation(sigmoid)

        self.fwrd_encs = [[EncoderLayer(d_model, num_heads, dff, dpo_rate) for _ in range(seq_len)] for _ in range(num_layers)]
        self.bcwrd_encs = [[EncoderLayer(d_model, num_heads, dff, dpo_rate) for _ in range()] for _ in range(num_layers)]

        self.mem = [[] for _ in range(num_layers)]
        self.mem_r = [[] for _ in range(num_layers)]

    def call(self, x, ex, cors, t_gate, training, mask=None):
        ex_enc = tf.expand_dims(self.dropout(self.ex_encoding(ex), training=training), axis=2)
        pos_enc = tf.expand_dims(spatial_posenc_batch(cors[..., 0], cors[..., 1], self.d_model))

        t_gate = self.sigm(self.gated_conv_t(t_gate, training))
        x = self.gated_conv(x, training) * t_gate
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x_shape = tf.shape(x)
        x = tf.reshape(x, [x_shape[0], x_shape[1], -1, x_shape[4]])
        x = x + ex_enc + pos_enc

        x_rev = tf.reverse(x, axis=1)

        self.mem[0] = [x[:, i, ...] for i in range(self.seq_len)]
        self.mem_r[0] = [x_rev[:, i, ...] for i in range(self.seq_len)]

        for i in range(self.num_layers):
            for l in range(self.seq_len - 1):
                inp = tf.concat([tf.stop_gradient(self.mem[i][l]), self.mem[i][l + 1]], axis=-1)
                inp_r = tf.concat([tf.stop_gradient(self.mem_r[i][l]), self.mem_r[i][l + 1]], axis=-1)
                output = self.fwrd_encs[i](inp, training, mask)
                output_r = self.bcwrd_encs[i](inp_r, training, mask)
                if i < len(self.mem):
                    self.mem[i + 1].append(output)
                    self.mem_r[i + 1].append(output_r)

        return x


class Decoder(layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, cnn_layers, cnn_filters, dpo_rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.ex_encoding = ex_encoding(d_model)
        self.dropout = layers.Dropout(dpo_rate)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, dpo_rate)
                           for _ in range(num_layers)]

    def call(self, x, ex, enc_output, training,
             look_ahead_mask=None, padding_mask=None):
        attention_weights = {}

        # ex_enc = self.ex_encoding(ex[:, :, :55])
        ex_enc = self.ex_encoding(ex)
        pos_enc = tf.expand_dims(tf.expand_dims(ex_enc, axis=1), axis=1)

        x = self.local_conv(x, training=training)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += pos_enc

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)
            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        return x, attention_weights


class ST_SAN(Model):
    def __init__(self, num_layers, d_model, num_heads, dff, cnn_layers, cnn_filters, seq_len,
                 d_final=256, output_size_t=4, dpo_rate=0.1):
        super(ST_SAN, self).__init__()

        self.encoder_f = Encoder(num_layers, d_model, num_heads, dff, cnn_layers, cnn_filters,
                                 seq_len, dpo_rate)
        self.encoder_t = Encoder(num_layers, d_model, num_heads, dff, cnn_layers, cnn_filters,
                                 seq_len, dpo_rate)

        self.decoder_f = Decoder(num_layers, d_model, num_heads, dff, cnn_layers, cnn_filters,
                                 dpo_rate)
        self.decoder_t = Decoder(num_layers, d_model, num_heads, dff, cnn_layers, cnn_filters,
                                 dpo_rate)

        self.dropout_t = layers.Dropout(dpo_rate)
        self.final_layer_t = layers.Dense(output_size_t, activation='tanh')

        self.gated_conv_1 = Gated_Conv_1(cnn_layers, cnn_filters, seq_len, dpo_rate=dpo_rate)
        self.gated_conv_2 = Gated_Conv_2(d_final, dpo_rate=dpo_rate)

    def call(self, flow_hist, trans_hist, ex_hist, flow_curr, trans_curr, ex_curr, training):
        flow_enc_inputs = tf.concat([flow_hist, flow_curr[:, :, :, 1:, :]], axis=-2)
        trans_enc_inputs = tf.concat([trans_hist, trans_curr[:, :, :, 1:, :]], axis=-2)
        ex_enc_inputs = tf.concat([ex_hist, ex_curr[:, 1:, :]], axis=-2)

        flow_dec_input = flow_curr[:, :, :, -1:, :]
        trans_dec_input = trans_curr[:, :, :, -1:, :]
        ex_dec_input = ex_curr[:, -1:, :]

        enc_outputs_t = self.encoder_t(trans_enc_inputs, ex_enc_inputs, training)
        enc_outputs_flow = self.encoder_f(flow_enc_inputs, ex_enc_inputs, training)

        enc_outputs_flow_gated = self.gated_conv_1(enc_outputs_flow, enc_outputs_t, training)

        dec_output_t, _ = self.decoder_t(trans_dec_input, ex_dec_input, enc_outputs_t, training)
        dec_output_flow, attention_weights_t = self.decoder_f(flow_dec_input, ex_dec_input, enc_outputs_flow_gated,
                                                              training)

        final_output_t = self.dropout_t(tf.squeeze(dec_output_t, axis=-2), training=training)
        final_output_t = self.final_layer_t(final_output_t)
        final_output = self.gated_conv_2(dec_output_flow, dec_output_t, training)

        return final_output, final_output_t, attention_weights_t
