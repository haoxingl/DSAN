from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from utils.DataLoader_Global import DataLoader_Global as dl


class DatasetGenerator:
    def __init__(self, dataset='taxi', batch_size=64, n_hist_week=0, n_hist_day=7,
                 n_hist_int=3, n_curr_int=1, n_int_before=1, n_pred=5):
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_hist_week = n_hist_week
        self.n_hist_day = n_hist_day
        self.n_hist_int = n_hist_int
        self.n_curr_int = n_curr_int
        self.n_int_before = n_int_before
        self.n_pred = n_pred
        self.train_data_loaded = False
        self.test_data_loaded = False
        self.data_loader = dl(self.dataset)

    def load_data(self, datatype, load_saved_data):
        inp_ft, inp_ex, dec_inp_f, dec_inp_t, dec_inp_ex, cors, y_t, y = self.data_loader.generate_data(
            datatype,
            self.n_hist_week,
            self.n_hist_day,
            self.n_hist_int,
            self.n_curr_int,
            self.n_int_before,
            self.n_pred,
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

    def build_dataset(self, datatype='train', load_saved_data=False, strategy=None):
        assert datatype == 'train' or datatype == 'test'
        if datatype == 'train':
            if not self.train_data_loaded:
                self.train_data_loaded = True

                self.train_dataset, data_shape = self.load_data(datatype, load_saved_data)

                self.data_size = int(data_shape[0])
                self.train_size = int(data_shape[0] * 0.8)

            dataset_cached = self.train_dataset.cache()
            dataset_shuffled = dataset_cached.shuffle(self.data_size, reshuffle_each_iteration=False)
            train_set = dataset_shuffled.take(self.train_size).shuffle(self.train_size)
            val_set = dataset_shuffled.skip(self.train_size)
            train_set = train_set.batch(self.batch_size).prefetch(tf.data.experimental.AUTOTUNE)
            val_set = val_set.batch(self.batch_size).prefetch(tf.data.experimental.AUTOTUNE)

            if strategy:
                return strategy.experimental_distribute_dataset(train_set), strategy.experimental_distribute_dataset(
                    val_set)
            else:
                return train_set, val_set

        else:
            if not self.test_data_loaded:
                self.test_data_loaded = True

                self.test_set, data_shape = self.load_data(datatype, load_saved_data)

                if self.batch_size > 1:
                    self.test_set = self.test_set.batch(self.batch_size)
                else:
                    self.test_set = self.test_set.shuffle(int(data_shape[0])).batch(self.batch_size)

            if strategy:
                return strategy.experimental_distribute_dataset(self.test_set)
            else:
                return self.test_set


def write_result(path, str):
    with open(path, 'a+') as file:
        file.write(str)


if __name__ == "__main__":
    dg = DatasetGenerator()
    a, b = dg.build_dataset(load_saved_data=True)
    c = dg.build_dataset(datatype='test', load_saved_data=True)
