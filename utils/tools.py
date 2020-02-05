from __future__ import absolute_import, division, print_function, unicode_literals

from utils.DataLoader import DataLoader as dl
import tensorflow as tf


class DatasetGenerator:
    def __init__(self, d_model=64, dataset='taxi', batch_size=64, n_hist_week=1, n_hist_day=3, n_hist_int=1,
                 n_curr_int=1, n_int_before=0, n_pred=6, local_block_len=3, local_block_len_g=5, test_model=False):
        self.d_model = d_model
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_hist_week = n_hist_week
        self.n_hist_day = n_hist_day
        self.n_hist_int = n_hist_int
        self.n_curr_int = n_curr_int
        self.n_int_before = n_int_before
        self.n_pred = n_pred
        self.local_block_len = local_block_len
        self.local_block_len_g = local_block_len_g
        self.test_model = test_model

    def load_data(self, datatype, st_revert=False, no_save=False, load_saved_data=False):
        data_loader = dl(self.d_model, self.dataset, self.local_block_len, self.local_block_len_g, self.test_model)
        inp_g, inp_l, inp_ex, dec_inp, dec_inp_ex, cors, cors_g, y = data_loader.generate_data(
            datatype,
            self.n_hist_week,
            self.n_hist_day,
            self.n_hist_int,
            self.n_curr_int,
            self.n_int_before,
            self.n_pred,
            st_revert,
            no_save,
            load_saved_data
        )

        dataset = tf.data.Dataset.from_tensor_slices(
            (
                {
                    "inp_g": inp_g,
                    "inp_l": inp_l,
                    "inp_ex": inp_ex,
                    "dec_inp": dec_inp,
                    "dec_inp_ex": dec_inp_ex,
                    "cors": cors,
                    "cors_g": cors_g
                },
                {
                    "y": y
                }
            )
        )

        return dataset, inp_g.shape[0]

    def build_dataset(self, datatype='train', load_saved_data=False, strategy=None, st_revert=False, no_save=None):
        assert datatype in ['train', 'val', 'test']

        dataset, data_size = self.load_data(datatype, st_revert, no_save, load_saved_data)

        if datatype == 'train':
            dataset_out = dataset.shuffle(data_size).batch(self.batch_size)\
                .cache().prefetch(tf.data.experimental.AUTOTUNE)
        elif datatype == 'val':
            dataset_out = dataset.batch(self.batch_size) \
                .cache().prefetch(tf.data.experimental.AUTOTUNE)
        else:
            dataset_out = dataset.batch(self.batch_size)

        return strategy.experimental_distribute_dataset(dataset_out) if strategy else dataset_out


def write_result(path, str, print_str=True):
    if print_str:
        print(str)
    with open(path, 'a+') as file:
        file.write(str + '\n')


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask

def create_padding_mask(inp_l):
    oup = tf.math.reduce_sum(inp_l, axis=-1)
    shape = tf.shape(oup)
    oup = tf.reshape(oup, [shape[0], shape[1], -1])
    mask = tf.cast(tf.math.equal(oup, 0), tf.float32)
    return mask

def create_padding_mask_tar(inp_l):
    oup = tf.math.reduce_sum(inp_l, axis=-1)
    mask = tf.cast(tf.math.equal(oup, 0), tf.float32)
    return mask


def create_masks(inp_g, inp_l, tar):
    padding_mask_g = create_padding_mask(inp_g)[:, :, tf.newaxis, tf.newaxis, :]
    padding_mask = create_padding_mask(inp_l)[:, :, tf.newaxis, tf.newaxis, :]

    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask_tar(tar)[:, tf.newaxis, tf.newaxis, tf.newaxis, :]
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return padding_mask_g, padding_mask, combined_mask

if __name__ == "__main__":
    dg = DatasetGenerator()
    a = dg.build_dataset(load_saved_data=True)
    b = dg.build_dataset(datatype='val', load_saved_data=True)
    c = dg.build_dataset(datatype='test', load_saved_data=True)
