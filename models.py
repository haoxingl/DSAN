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


class GatedConv(layers.Layer):
    def __init__(self, num_layers, num_filters, seq_len, dpo_rate=0.1, sig_act=False):
        super(GatedConv, self).__init__()

        self.seq_len = seq_len  # indicate how many time intervals are included in the historical inputs
        self.num_layers = num_layers
        self.sig_act = sig_act

        self.convs = [[layers.Conv2D(num_filters, (3, 3), activation=actfunc, padding='same')
                       for _ in range(num_layers)] for _ in range(seq_len)]
        self.dpo_layers = [layers.Dropout(dpo_rate) for _ in range(seq_len)]

        if sig_act:
            self.sigm = layers.Activation(sigmoid)

    def call(self, inp, training):
        outputs = []

        inputs = tf.split(inp, self.seq_len, axis=1)
        for i in range(self.seq_len):
            output = tf.squeeze(inputs[i], axis=1)
            for j in range(self.num_layers):
                output = self.convs[i][j](output)
            output = self.dpo_layers[i](output, training=training)
            output = tf.expand_dims(output, axis=1)
            outputs.append(output)

        output_final = self.sigm(tf.concat(outputs, axis=1)) if self.sig_act else tf.concat(outputs, axis=1)

        return output_final


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
    def __init__(self, d_model, num_heads, seq_len):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.seq_len = seq_len

        assert d_model % self.num_heads == 0

        self.depth = seq_len * d_model // self.num_heads

        self.wq = layers.Dense(d_model * seq_len)
        self.wk = layers.Dense(d_model * seq_len)
        self.wv = layers.Dense(d_model * seq_len)

        self.dense = layers.Dense(d_model * seq_len)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)
        q = self.split_heads(q, batch_size)
        q = tf.split(q, self.seq_len, axis=-1)

        k = self.wk(k)
        k = self.split_heads(k, batch_size)
        k = tf.split(k, self.seq_len, axis=-1)

        v = self.wv(v)
        v = self.split_heads(v, batch_size)
        v = tf.split(v, self.seq_len, axis=-1)

        outputs = []
        weights = []

        for i in range(self.seq_len):
            scaled_attention, attention_weights = scaled_dot_product_attention(q[i], k[i], v[i], mask)

            scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

            concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

            outputs.append(self.dense[i](concat_attention))
            weights.append(attention_weights)

        output = tf.concat(outputs, axis=-1)
        output = self.dense(output)

        return output, weights


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
    def __init__(self, d_model, num_heads, dff, seq_len, dpo_rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads, seq_len)
        self.ffn = point_wise_feed_forward_network(d_model * seq_len, dff * seq_len)

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
    def __init__(self, d_model, num_heads, dff, seq_len, dpo_rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads, seq_len)
        self.mha2 = MultiHeadAttention(d_model, num_heads, seq_len)

        self.ffn = point_wise_feed_forward_network(d_model * seq_len, dff * seq_len)

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
    def __init__(self, num_layers, d_model, d_global, num_heads, dff, cnn_layers, cnn_filters, seq_len, dpo_rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.seq_len = seq_len

        self.ex_encoder = ex_encoding(d_model)
        self.dropout = layers.Dropout(dpo_rate)

        self.gated_conv = GatedConv(cnn_layers, cnn_filters, seq_len, dpo_rate)
        self.gated_conv_t = GatedConv(cnn_layers, cnn_filters, seq_len, dpo_rate, sig_act=True)

        self.encs = [EncoderLayer(d_model, num_heads, dff, seq_len, dpo_rate) for _ in range(num_layers)]

    def call(self, x, ex, cors, t_gate, training, mask=None):
        data_shape = tf.shape(x)

        ex_enc = tf.expand_dims(self.ex_encoder(ex), axis=2)
        pos_enc = tf.expand_dims(cors, axis=1)

        x_gated = self.gated_conv(x, training) * self.gated_conv_t(t_gate, training)
        x_gated *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x_flat = tf.reshape(x_gated, [data_shape[0], data_shape[1], -1, self.d_model])
        enc_inp = x_flat + ex_enc + pos_enc

        enc_inp = self.dropout(enc_inp, training=training)

        mem = tf.split(enc_inp, self.seq_len, axis=1)

        for i, tensor in enumerate(mem):
            mem[i] = tf.squeeze(tensor, axis=1)

        mem = tf.concat(mem, axis=-1)

        for i in range(self.num_layers):
            mem = self.encs[i](mem, training, mask)

        return mem


class Decoder(layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, seq_len, dpo_rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.seq_len = seq_len

        self.ex_encoder = ex_encoding(d_model)
        self.dropout = layers.Dropout(dpo_rate)

        self.li_conv = Sequential([layers.Dense(d_model, activation=actfunc) for _ in range(3)])

        self.decs = [DecoderLayer(d_model, num_heads, dff, seq_len, dpo_rate) for _ in range(num_layers)]
        self.out_lyr = layers.Dense(d_model, activation=actfunc)
        self.dropout_out = layers.Dropout(dpo_rate)

    def call(self, x, ex, enc_outputs, training, concated=True, look_ahead_mask=None, padding_mask=None):
        attention_weights = {}

        ex_enc = self.ex_encoder(ex)
        pos_enc = spatial_posenc(0, 0, self.d_model)

        x_conved = self.li_conv(x)
        x_conved *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x_coded = x_conved + ex_enc + pos_enc

        x_coded = self.dropout(x_coded, training=training)

        mem = [x_coded for _ in range(self.seq_len)]
        mem = tf.concat(mem, axis=-1)
        enc_output = tf.concat(enc_outputs, axis=-1)

        for i in range(self.num_layers):
            mem, block1, block2 = self.decs[i](mem, enc_output, training, look_ahead_mask, padding_mask)
            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        if concated:
            output = tf.concat(mem, axis=-1)
            output = self.out_lyr(output)
            output = self.dropout_out(output, training=training)
        else:
            output = mem

        return output, attention_weights


class STSAN_XL(Model):
    def __init__(self, num_layers, d_model, d_global, num_heads, dff, cnn_layers, cnn_filters, seq_len, dpo_rate=0.1):
        super(STSAN_XL, self).__init__()

        self.encoder = Encoder(num_layers, d_model, d_global, num_heads, dff, cnn_layers, cnn_filters, seq_len,
                               dpo_rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff, seq_len, dpo_rate)

        self.final_lyr = layers.Dense(2, activation='tanh')

    def call(self, inp_ft, inp_ex, dec_inp_f, dec_inp_ex, cors, training, look_ahead_mask=None):
        inp_t, t_gate = tf.split(inp_ft, [2, 4], axis=-1)

        enc_outputs = self.encoder(inp_t, inp_ex, cors, t_gate, training)

        dec_outputs, attention_weights = \
            self.decoder(dec_inp_f, dec_inp_ex, enc_outputs, training, True, look_ahead_mask=look_ahead_mask)

        final_output = self.final_lyr(dec_outputs)

        return final_output, attention_weights
