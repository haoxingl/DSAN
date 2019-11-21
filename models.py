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


class GatedConv(layers.Layer):
    def __init__(self, num_layers, num_filters, seq_len, dpo_rate=0.1, sig_act=False):
        super(GatedConv, self).__init__()

        self.seq_len = seq_len  # indicate how many time intervals are included in the historical inputs
        self.num_layers = num_layers
        self.sig_act = sig_act

        self.convs = [[layers.Conv2D(num_filters, (3, 3), activation=actfunc, padding='same')
                               for _ in range(num_layers)] for _ in range(seq_len)]
        self.dpo_layers = [layers.Dropout(dpo_rate * num_layers) for _ in range(seq_len)]

        if sig_act:
            self.sigm = layers.Activation(sigmoid)

    def call(self, inp, training):
        outputs = []

        for i in range(self.seq_len):
            output = inp[:, i, ...]
            for j in range(self.num_layers):
                output = self.convs[i][j](output)
            output = self.dpo_layers[i](output, training=training)
            output = tf.expand_dims(output, axis=1)
            outputs.append(output)

        output_final = self.sigm(tf.concat(outputs, axis=1))

        return output_final
    
    
def LinearConv(d_model, dpo_rate=0.1):
    return Sequential([layers.Dense(d_model, activation=actfunc) for _ in range(2)]
                      + [DenseDropout(d_model, dpo_rate)])


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


class DenseDropout(layers.Layer):
    def __init__(self, out_size, dpo_rate=0.1):
        super(DenseDropout, self).__init__()

        self.dense = layers.Dense(out_size, activation=actfunc)
        self.dropout = layers.Dropout(dpo_rate)

    def call(self, x, training, dpo_out=True):
        if dpo_out:
            return self.dropout(self.dense(x), training=training)
        else:
            return self.dense(self.dropout(x, training=training))


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
    def __init__(self, num_layers, d_model, d_global, num_heads, dff, cnn_layers, cnn_filters, seq_len, dpo_rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.seq_len = seq_len

        self.ex_enc = ex_encoding(d_model)
        self.dropout = layers.Dropout(dpo_rate)

        self.gated_conv = GatedConv(cnn_layers, cnn_filters, seq_len, dpo_rate)
        self.gated_conv_t = GatedConv(cnn_layers, cnn_filters, seq_len, dpo_rate, sig_act=True)

        self.encs = [[EncoderLayer(d_model, num_heads, dff, dpo_rate) for _ in range(seq_len)] for _ in range(num_layers)]
        self.dense_g = [[DenseDropout(d_global, dpo_rate) for _ in range(seq_len)] for _ in range(num_layers)]

        self.mem = [[] for _ in range(num_layers + 1)]

    def call(self, x, ex, cors, t_gate, training, mask=None):
        data_shape = tf.shape(x)

        ex_enc = tf.expand_dims(self.dropout(self.ex_encoding(ex), training=training), axis=2)
        pos_enc = tf.expand_dims(spatial_posenc_batch(cors[..., 0], cors[..., 1], self.d_model), axis=1)

        x = self.gated_conv(x, training) * self.gated_conv_t(t_gate, training)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = tf.reshape(x, [data_shape[0], data_shape[1], -1, data_shape[4]])
        x += ex_enc + pos_enc

        self.mem[0] = [x[:, i, ...] for i in range(self.seq_len)]

        for i in range(self.num_layers):
            for l in range(self.seq_len):
                inp_g = tf.concat([tf.stop_gradient(self.mem[i][k]) for k in range(self.seq_len) if k != l], axis=-1)
                inp_g = self.dense_g[i][l](inp_g, training)
                inp = tf.concat([self.mem[i][l], inp_g], axis=-1)
                output = self.encs[i][l](inp, training, mask)
                self.mem[i + 1].append(output)

        return self.mem[self.num_layers]


class Decoder(layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, seq_len, dpo_rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.seq_len = seq_len

        self.ex_encoding = ex_encoding(d_model)
        self.dpo_ex = layers.Dropout(dpo_rate)

        self.li_conv = LinearConv(d_model, dpo_rate)

        self.decs = [[DecoderLayer(d_model, num_heads, dff, dpo_rate) for _ in range(seq_len)] for _ in range(num_layers)]
        self.out_lyr = [layers.Dense(2, activation=actfunc) for _ in range(seq_len)]

    def call(self, x, ex, mem, training,
             look_ahead_mask=None, padding_mask=None):
        attention_weights = {}

        ex_enc = self.dpo_ex(self.ex_encoding(ex), training=training)
        pos_enc = spatial_posenc(0, 0, self.d_model)

        x = self.li_conv(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + ex_enc + pos_enc

        outputs = [x for _ in range(self.seq_len)]

        for i in range(self.num_layers):
            for l in range(self.seq_len):
                outputs[l], block1, block2 = self.dec_layers[i](x, mem[l], training, look_ahead_mask, padding_mask)
                attention_weights['decoder{}_layer{}_block1'.format(l + 1, i + 1)] = block1
                attention_weights['decoder{}_layer{}_block2'.format(l + 1, i + 1)] = block2
                outputs[l] = x

        return outputs, attention_weights


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
