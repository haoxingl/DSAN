from __future__ import absolute_import, division, print_function, unicode_literals

from utils.DataLoader import DataLoader as dl
import tensorflow as tf


class DatasetGenerator:
    def __init__(self, d_model=64, dataset='taxi', batch_size=64, n_hist_week=0, n_hist_day=7, n_hist_int=3,
                 n_curr_int=1, n_int_before=1, n_pred=5, local_block_len=None, test_model=False):
        self.d_model = d_model
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_hist_week = n_hist_week
        self.n_hist_day = n_hist_day
        self.n_hist_int = n_hist_int
        self.n_curr_int = n_curr_int
        self.n_int_before = n_int_before
        self.n_pred = n_pred
        self.test_model = test_model
        self.local_block_len = local_block_len
        self.train_data_loaded = False
        self.test_data_loaded = False

    def load_data(self, datatype, no_save=False, load_saved_data=False):
        data_loader = dl(self.d_model, self.dataset, self.local_block_len, self.test_model)
        inp_ft, inp_ex, dec_inp_f, dec_inp_t, dec_inp_ex, cors, y_t, y = data_loader.generate_data(
            datatype,
            self.n_hist_week,
            self.n_hist_day,
            self.n_hist_int,
            self.n_curr_int,
            self.n_int_before,
            self.n_pred,
            self.local_block_len,
            no_save,
            load_saved_data
        )

        dataset = tf.data.Dataset.from_tensor_slices(
            (
                {
                    "inp_ft": inp_ft,
                    "inp_ex": inp_ex,
                    "dec_inp_f": dec_inp_f,
                    "dec_inp_t": dec_inp_t,
                    "dec_inp_ex": dec_inp_ex,
                    "cors": cors
                },
                {
                    "y_t": y_t,
                    "y": y
                }
            )
        )

        return dataset, inp_ft.shape

    def build_dataset(self, datatype='train', load_saved_data=False, strategy=None, no_save=None):
        assert datatype == 'train' or datatype == 'test'
        if datatype == 'train':
            train_dataset, data_shape = self.load_data(datatype, no_save, load_saved_data or self.train_data_loaded)

            if not self.train_data_loaded:
                self.train_data_loaded = True

            data_size = int(data_shape[0])
            train_size = int(data_shape[0] * 0.8)

            dataset_cached = train_dataset.cache()
            dataset_shuffled = dataset_cached.shuffle(data_size, reshuffle_each_iteration=False)
            train_set = dataset_shuffled.take(train_size).shuffle(train_size)
            val_set = dataset_shuffled.skip(train_size)
            train_set = train_set.batch(self.batch_size)
            val_set = val_set.batch(self.batch_size)
            # train_set = train_set.batch(self.batch_size).prefetch(tf.data.experimental.AUTOTUNE)
            # val_set = val_set.batch(self.batch_size).prefetch(tf.data.experimental.AUTOTUNE)

            if strategy:
                return strategy.experimental_distribute_dataset(train_set), strategy.experimental_distribute_dataset(
                    val_set)
            else:
                return train_set, val_set

        else:
            if not self.test_data_loaded:
                self.test_data_loaded = True

                test_set, data_shape = self.load_data(datatype, no_save, load_saved_data)

                if self.batch_size > 1:
                    test_set = test_set.batch(self.batch_size)
                else:
                    test_set = test_set.shuffle(int(data_shape[0])).batch(self.batch_size)

            if strategy:
                return strategy.experimental_distribute_dataset(test_set)
            else:
                return test_set


def write_result(path, str, print_str=True):
    if print_str:
        print(str)
    with open(path, 'a+') as file:
        file.write(str + '\n')


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask


def create_padding_mask(inp, dec_inp=False):
    inp = tf.math.reduce_sum(inp, axis=-1)
    if dec_inp:
        inp = inp[:, tf.newaxis, tf.newaxis, tf.newaxis, :]
    else:
        shape = tf.shape(inp)
        inp = tf.reshape(inp, [shape[0], shape[1], -1])
        inp = inp[:, :, tf.newaxis, tf.newaxis, :]
    inp = tf.cast(tf.math.equal(inp, 0), tf.float32)
    return inp


def create_padding_mask_t(inp):
    inp = tf.math.reduce_sum(inp, [2, 3, 4])
    inp = tf.cast(tf.math.equal(inp, 0), tf.float32)
    return inp[:, tf.newaxis, tf.newaxis, tf.newaxis, :]


def create_masks(inp, tar):
    enc_padding_mask = create_padding_mask(inp)
    dec_padding_mask = create_padding_mask(inp)
    dec_padding_mask_t = create_padding_mask(tar, dec_inp=True)
    dec_padding_mask_t = tf.transpose(dec_padding_mask_t, perm=[0, 4, 1, 2, 3])

    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar, dec_inp=True)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask, dec_padding_mask_t


if __name__ == "__main__":
    dg = DatasetGenerator(local_block_len=3, test_model=100)
    a, b = dg.build_dataset(no_save=True)
    c = dg.build_dataset(datatype='test', no_save=True)
