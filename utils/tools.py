from __future__ import absolute_import, division, print_function, unicode_literals

from utils.DataLoader import DataLoader as dl
import tensorflow as tf


class DatasetGenerator:
    def __init__(self, d_model=64, dataset='taxi', batch_size=64, n_w=1, n_d=3, n_wd_times=1, n_p=1, n_before=0,
                 n_pred=12, l_half=3, l_half_g=None, pre_shuffle=True, same_padding=False, test_model=False):
        self.d_model = d_model
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_w = n_w
        self.n_d = n_d
        self.n_wd_times = n_wd_times
        self.n_p = n_p
        self.n_before = n_before
        self.n_pred = n_pred
        self.l_half = l_half
        self.l_half_g = l_half_g
        self.pre_shuffle = pre_shuffle
        self.same_padding = same_padding
        self.test_model = test_model

        self.val_set = None

    def load_data(self, datatype, load_saved_data=False, st_revert=False, no_save=False):
        data_loader = dl(self.d_model, self.dataset, self.l_half, self.l_half_g,
                         self.pre_shuffle, self.same_padding, self.test_model)
        dae_inp_g, dae_inp, dae_inp_ex, sad_inp, sad_inp_ex, cors, cors_g, y = data_loader.generate_data(
            datatype, self.n_w, self.n_d, self.n_wd_times, self.n_p, self.n_before, self.n_pred, load_saved_data,
            st_revert, no_save)

        if self.pre_shuffle and datatype == 'train':
            train_set = tf.data.Dataset.from_tensor_slices(
                (
                    {
                        "dae_inp_g": dae_inp_g[0],
                        "dae_inp": dae_inp[0],
                        "dae_inp_ex": dae_inp_ex[0],
                        "sad_inp": sad_inp[0],
                        "sad_inp_ex": sad_inp_ex[0],
                        "cors": cors[0],
                        "cors_g": cors_g[0]
                    },
                    {
                        "y": y[0]
                    }
                )
            )

            val_set = tf.data.Dataset.from_tensor_slices(
                (
                    {
                        "dae_inp_g": dae_inp_g[1],
                        "dae_inp": dae_inp[1],
                        "dae_inp_ex": dae_inp_ex[1],
                        "sad_inp": sad_inp[1],
                        "sad_inp_ex": sad_inp_ex[1],
                        "cors": cors[1],
                        "cors_g": cors_g[1]
                    },
                    {
                        "y": y[1]
                    }
                )
            )

            return [train_set, val_set], dae_inp_g[0].shape[0]
        else:
            dataset = tf.data.Dataset.from_tensor_slices(
                (
                    {
                        "dae_inp_g": dae_inp_g,
                        "dae_inp": dae_inp,
                        "dae_inp_ex": dae_inp_ex,
                        "sad_inp": sad_inp,
                        "sad_inp_ex": sad_inp_ex,
                        "cors": cors,
                        "cors_g": cors_g
                    },
                    {
                        "y": y
                    }
                )
            )

            return dataset, dae_inp_g.shape[0]

    def build_dataset(self, datatype='train', load_saved_data=False, strategy=None, st_revert=False, no_save=None):
        assert datatype in ['train', 'val', 'test']

        if datatype == 'val' and self.pre_shuffle:
            assert self.val_set
            pass
        else:
            dataset, data_size = self.load_data(datatype, load_saved_data, st_revert, no_save)

        if datatype == 'train':
            if not self.pre_shuffle:
                dataset_out = dataset.shuffle(data_size).batch(self.batch_size).cache().prefetch(
                    tf.data.experimental.AUTOTUNE)
            else:
                self.val_set = dataset[1]
                dataset_out = dataset[0].shuffle(data_size).batch(self.batch_size).cache().prefetch(
                    tf.data.experimental.AUTOTUNE)
        elif datatype == 'val':
            if not self.pre_shuffle:
                dataset_out = dataset.batch(self.batch_size) \
                    .cache().prefetch(tf.data.experimental.AUTOTUNE)
            else:
                dataset_out = self.val_set.batch(self.batch_size) \
                    .cache().prefetch(tf.data.experimental.AUTOTUNE)
        else:
            if self.batch_size == 1:
                dataset_out = dataset.shuffle(data_size).batch(self.batch_size)
            else:
                dataset_out = dataset.batch(self.batch_size)

        return strategy.experimental_distribute_dataset(dataset_out) if strategy else dataset_out


class ResultWriter:
    def __init__(self, path):
        self.path = path

    def write(self, str, print_str=True):
        if print_str:
            print(str)
        with open(self.path, 'a+') as file:
            file.write(str + '\n')


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask


def create_padding_mask(inp):
    oup = tf.math.reduce_sum(inp, axis=-1)
    shape = tf.shape(oup)
    oup = tf.reshape(oup, [shape[0], shape[1], -1])
    mask = tf.cast(tf.math.equal(oup, 0), tf.float32)
    return mask


def create_padding_mask_tar(inp):
    oup = tf.math.reduce_sum(inp, axis=-1)
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
