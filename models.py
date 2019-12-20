from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential, backend
from tensorflow.keras.utils import get_custom_objects

def gelu(x):
    return 0.5 * x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))

get_custom_objects().update({'gelu': layers.Activation(gelu)})

actfunc = 'relu'


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
    def __init__(self, num_layers, num_filters, seq_len, dpo_rate=0.1):
        super(GatedConv, self).__init__()

        self.seq_len = seq_len  # indicate how many time intervals are included in the historical inputs
        self.num_layers = num_layers

        self.convs = [[layers.Conv2D(num_filters, (3, 3), activation=actfunc, padding='same')
                       for _ in range(num_layers)] for _ in range(seq_len)]
        self.dpo_layers = [[layers.Dropout(dpo_rate) for _ in range(num_layers)] for _ in range(seq_len)]

    def call(self, inp, training):
        outputs = []
        inputs = tf.split(inp, self.seq_len, axis=1)
        for i in range(self.seq_len):
            output = tf.squeeze(inputs[i], axis=1)
            for j in range(self.num_layers):
                output = self.convs[i][j](output)
                output = self.dpo_layers[i][j](output, training=training)
            output = tf.expand_dims(output, axis=1)
            outputs.append(output)

        output_final = tf.concat(outputs, axis=1)

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
    def __init__(self, d_model, num_heads, self_all=True):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.self_all = self_all

        assert d_model % num_heads == 0

        self.depth = d_model // num_heads

        if self_all:
            self.wx = layers.Dense(d_model * 3)
        else:
            self.wq = layers.Dense(d_model)
            self.wkv = layers.Dense(d_model * 2)

        self.dense = layers.Dense(d_model)

    def split_heads(self, x):
        shape = tf.shape(x)
        x = tf.reshape(x, (shape[0], shape[1], shape[2], self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 1, 3, 2, 4])

    def call(self, v, k, q, mask):
        if self.self_all:
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

        output = self.dense(concat_attention)

        return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    return Sequential([
        layers.Dense(dff, activation=actfunc),
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
    def __init__(self, d_model, num_heads, dff, dpo_rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads, self_all=False)

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
        out3 = self.layernorm3(ffn_output + out2)

        return out3, attn_weights_block1, attn_weights_block2


class DecoderLayer_NLAM(layers.Layer):
    def __init__(self, d_model, num_heads, dff, dpo_rate=0.1):
        super(DecoderLayer_NLAM, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads, self_all=False)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(dpo_rate)
        self.dropout2 = layers.Dropout(dpo_rate)

    def call(self, x, enc_output_x, training, padding_mask):
        attn1, attn_weights_block = self.mha(enc_output_x, enc_output_x, x, padding_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2, attn_weights_block


class Encoder(layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, cnn_layers, cnn_filters, seq_len, dpo_rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.ex_encoder = ex_encoding(d_model, dff)
        self.dropout = layers.Dropout(dpo_rate)

        self.gated_conv = GatedConv(cnn_layers, cnn_filters, seq_len, dpo_rate)

        self.encs = [EncoderLayer(d_model, num_heads, dff, dpo_rate) for _ in range(num_layers)]

    def call(self, x, ex, cors, training, mask):
        shape = tf.shape(x)

        ex_enc = tf.expand_dims(self.ex_encoder(ex), axis=2)
        pos_enc = tf.expand_dims(cors, axis=1)

        x_gated = self.gated_conv(x, training)
        x_gated *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x_flat = tf.reshape(x_gated, [shape[0], shape[1], -1, self.d_model])
        enc_inp = x_flat + ex_enc + pos_enc
        # enc_inp = x_flat + ex_enc

        output = self.dropout(enc_inp, training=training)

        for i in range(self.num_layers):
            output = self.encs[i](output, training, mask)

        return output


class Decoder(layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, dpo_rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.ex_encoder = ex_encoding(d_model, dff)
        self.dropout = layers.Dropout(dpo_rate)

        self.li_conv = Sequential([layers.Dense(d_model, activation=actfunc) for _ in range(3)])

        self.decs_s = [DecoderLayer(d_model, num_heads, dff, dpo_rate) for _ in range(num_layers)]
        self.decs_t = [DecoderLayer(d_model, num_heads, dff, dpo_rate) for _ in range(num_layers)]
        self.dropout_out = layers.Dropout(dpo_rate)

    def call(self, x, ex, enc_output, training, look_ahead_mask, padding_mask, padding_mask_t):
        attention_weights = {}

        ex_enc = self.ex_encoder(ex)
        pos_enc = spatial_posenc(0, 0, self.d_model)

        x_conved = self.li_conv(x)
        x_conved *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x_coded = x_conved + ex_enc + pos_enc
        # x_coded = x_conved + ex_enc

        x_coded = self.dropout(x_coded, training=training)
        dec_output_s = tf.expand_dims(x_coded, axis=1)
        dec_output_t = tf.transpose(dec_output_s, perm=[0, 2, 1, 3])

        for i in range(self.num_layers):
            dec_output_s, block1, block2 = self.decs_s[i](dec_output_s, enc_output, training, look_ahead_mask,
                                                          padding_mask)
            attention_weights['decoder_s_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_s_layer{}_block2'.format(i + 1)] = block2

        dec_output_s = tf.transpose(dec_output_s, perm=[0, 2, 1, 3])

        for i in range(self.num_layers):
            dec_output_t, block1, block2 = self.decs_t[i](dec_output_t, dec_output_s, training, padding_mask_t, None)
            attention_weights['decoder_t_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_t_layer{}_block2'.format(i + 1)] = block2

        dec_output = self.dropout_out(tf.squeeze(dec_output_t, axis=-2), training=training)

        return dec_output, attention_weights


class STSAN_XL(Model):
    def __init__(self, num_layers, d_model, num_heads, dff, cnn_layers, cnn_filters, seq_len, dpo_rate=0.1):
        super(STSAN_XL, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff, cnn_layers, cnn_filters, seq_len, dpo_rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff, dpo_rate)

        self.final_lyr = layers.Dense(2, activation='tanh')

    def call(self, inp_ft, inp_ex, dec_inp_f, dec_inp_ex, cors, training,
             enc_padding_mask, look_ahead_mask, dec_padding_mask, dec_padding_mask_t):
        enc_output = self.encoder(inp_ft, inp_ex, cors, training, enc_padding_mask)

        dec_output, attention_weights = \
            self.decoder(dec_inp_f, dec_inp_ex, enc_output, training,
                         look_ahead_mask, dec_padding_mask, dec_padding_mask_t)

        final_output = self.final_lyr(dec_output)

        return final_output, attention_weights
