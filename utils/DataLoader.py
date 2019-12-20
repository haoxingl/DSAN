import numpy as np
from utils.CordinateGenerator import CordinateGenerator

import parameters_nyctaxi as param_taxi
import parameters_nycbike as param_bike


class DataLoader:
    def __init__(self, d_model, dataset='taxi', local_block_len=None, test_model=False):
        assert dataset == 'taxi' or 'bike'
        self.dataset = dataset
        self.pmt = param_taxi if dataset == 'taxi' else param_bike
        self.cor_gen = CordinateGenerator(self.pmt.len_r, self.pmt.len_c, d_model, local_block_len=local_block_len)
        self.test_model = test_model

    def load_data_f(self, datatype='train'):
        if datatype == 'train':
            self.f_train = np.array(np.load(self.pmt.f_train)['data'], dtype=np.float32) / self.pmt.f_train_max
        else:
            self.f_test = np.array(np.load(self.pmt.f_test)['data'], dtype=np.float32) / self.pmt.f_train_max

    def load_data_t(self, datatype='train'):
        if datatype == 'train':
            self.t_train = np.array(np.load(self.pmt.t_train)['data'], dtype=np.float32) / self.pmt.t_train_max
        else:
            self.t_test = np.array(np.load(self.pmt.t_test)['data'], dtype=np.float32) / self.pmt.t_train_max

    """ external_knowledge contains the time and weather information of each time interval """

    def load_data_ex(self, datatype='train'):
        if datatype == 'train':
            self.ex_train = np.load(self.pmt.ex_train)['data']
        else:
            self.ex_test = np.load(self.pmt.ex_test)['data']

    def generate_data(self, datatype='train',
                      n_hist_week=0,  # number previous weeks we generate the sample from.
                      n_hist_day=7,  # number of the previous days we generate the sample from
                      n_hist_int=3,  # number of intervals we sample in the previous weeks, days
                      n_curr_int=1,  # number of intervals we sample in the current day
                      n_int_before=1,  # number of intervals before the predicted interval
                      n_pred=5,
                      local_block_len=None,
                      no_save=False,
                      load_saved_data=False):  # loading the previous saved data

        assert datatype == 'train' or datatype == 'test'

        """ loading saved data """
        if load_saved_data and not self.test_model:
            print('Loading {} data from .npzs...'.format(datatype))
            inp_ft = np.load("data/inp_ft_{}_{}.npz".format(self.dataset, datatype))['data']
            inp_ex = np.load("data/inp_ex_{}_{}.npz".format(self.dataset, datatype))['data']
            dec_inp_f = np.load("data/dec_inp_f_{}_{}.npz".format(self.dataset, datatype))['data']
            dec_inp_t = np.load("data/dec_inp_t_{}_{}.npz".format(self.dataset, datatype))['data']
            dec_inp_ex = np.load("data/dec_inp_ex_{}_{}.npz".format(self.dataset, datatype))['data']
            cors = np.load("data/cors_{}_{}.npz".format(self.dataset, datatype))['data']
            y_t = np.load("data/y_t_{}_{}.npz".format(self.dataset, datatype))['data']
            y = np.load("data/y_{}_{}.npz".format(self.dataset, datatype))['data']

            return inp_ft, inp_ex, dec_inp_f, dec_inp_t, dec_inp_ex, cors, y_t, y
        else:
            print("Loading {} data...".format(datatype))
            """ loading data """
            self.load_data_f(datatype)
            self.load_data_t(datatype)
            self.load_data_ex(datatype)
            if local_block_len:
                block_full_len = 2 * local_block_len + 1

            if datatype == "train":
                f_data = self.f_train
                t_data = self.t_train
                ex_data = self.ex_train
            elif datatype == "test":
                f_data = self.f_test
                t_data = self.t_test
                ex_data = self.ex_test
            else:
                print("Please select **train** or **test**")
                raise Exception

            """ initialize the array to hold the final inputs """

            inp_ft = []
            inp_ex = []

            dec_inp_f = []
            dec_inp_t = []
            dec_inp_ex = []

            cors = []

            y = []  # ground truth of the inflow and outflow of each node at each time interval
            y_t = []  # ground truth of the transitions between each node and its neighbors in the area of interest

            assert n_hist_week >= 0 and n_hist_day >= 1
            """ set the start time interval to sample the data"""
            s1 = n_hist_day * self.pmt.n_int_day + n_int_before
            s2 = n_hist_week * 7 * self.pmt.n_int_day + n_int_before
            time_start = max(s1, s2)
            time_end = f_data.shape[0] - n_pred

            data_shape = f_data.shape

            for t in range(time_start, time_end):
                if t % 100 == 0:
                    print("Currently at {} interval...".format(t))

                for r in range(data_shape[1]):
                    for c in range(data_shape[2]):

                        """ initialize the array to hold the samples of each node at each time interval """

                        inp_ft_sample = []
                        inp_ex_sample = []

                        if local_block_len:
                            """ initialize the boundaries of the area of interest """
                            r_start = r - local_block_len  # the start location of each AoI
                            c_start = c - local_block_len

                            """ adjust the start location if it is on the boundaries of the grid map """
                            if r_start < 0:
                                r_start_local = 0 - r_start
                                r_start = 0
                            else:
                                r_start_local = 0
                            if c_start < 0:
                                c_start_local = 0 - c_start
                                c_start = 0
                            else:
                                c_start_local = 0

                            r_end = r + local_block_len + 1  # the end location of each AoI
                            c_end = c + local_block_len + 1
                            if r_end >= data_shape[1]:
                                r_end_local = block_full_len - (r_end - data_shape[1])
                                r_end = data_shape[1]
                            else:
                                r_end_local = block_full_len
                            if c_end >= data_shape[2]:
                                c_end_local = block_full_len - (c_end - data_shape[2])
                                c_end = data_shape[2]
                            else:
                                c_end_local = block_full_len

                        """ start the samplings of previous weeks """
                        for week_cnt in range(n_hist_week):
                            s_time_w = int(t - (n_hist_week - week_cnt) * 7 * self.pmt.n_int_day - n_int_before)

                            for int_cnt in range(n_hist_int):
                                t_now = s_time_w + int_cnt

                                if not local_block_len:
                                    sample_f = f_data[t_now, ...]

                                    sample_t = np.zeros((data_shape[1], data_shape[2], 2), dtype=np.float32)

                                    sample_t[..., 0] += t_data[0, t_now, ..., r, c]
                                    sample_t[..., 0] += t_data[1, t_now, ..., r, c]
                                    sample_t[..., 1] += t_data[0, t_now, r, c, ...]
                                    sample_t[..., 1] += t_data[1, t_now, r, c, ...]

                                else:
                                    sample_f = np.zeros((block_full_len, block_full_len, 2), dtype=np.float32)
                                    sample_f[r_start_local:r_end_local, c_start_local:c_end_local, :] = f_data[
                                                                                                        t_now,
                                                                                                        r_start:r_end,
                                                                                                        c_start:c_end,
                                                                                                        :]

                                    sample_t = np.zeros((block_full_len, block_full_len, 2), dtype=np.float32)
                                    sample_t[r_start_local:r_end_local, c_start_local:c_end_local, 0] += \
                                        t_data[0, t_now, r_start:r_end, c_start:c_end, r, c]
                                    sample_t[r_start_local:r_end_local, c_start_local:c_end_local, 0] += \
                                        t_data[1, t_now, r_start:r_end, c_start:c_end, r, c]
                                    sample_t[r_start_local:r_end_local, c_start_local:c_end_local, 1] += \
                                        t_data[0, t_now, r, c, r_start:r_end, c_start:c_end]
                                    sample_t[r_start_local:r_end_local, c_start_local:c_end_local, 1] += \
                                        t_data[1, t_now, r, c, r_start:r_end, c_start:c_end]

                                inp_ft_sample.append(np.concatenate([sample_f, sample_t], axis=-1))
                                inp_ex_sample.append(ex_data[t_now, :])

                        """ start the samplings of previous days"""
                        for hist_day_cnt in range(n_hist_day):
                            """ define the start time in previous days """
                            s_time_d = int(t - (n_hist_day - hist_day_cnt) * self.pmt.n_int_day - n_int_before)

                            """ generate samples from the previous days """
                            for int_cnt in range(n_hist_int):
                                t_now = s_time_d + int_cnt

                                if not local_block_len:
                                    sample_f = f_data[t_now, ...]

                                    sample_t = np.zeros((data_shape[1], data_shape[2], 2), dtype=np.float32)

                                    sample_t[..., 0] += t_data[0, t_now, ..., r, c]
                                    sample_t[..., 0] += t_data[1, t_now, ..., r, c]
                                    sample_t[..., 1] += t_data[0, t_now, r, c, ...]
                                    sample_t[..., 1] += t_data[1, t_now, r, c, ...]

                                else:
                                    # define the matrix to hold the historical flow inputs of AoI
                                    sample_f = np.zeros((block_full_len, block_full_len, 2), dtype=np.float32)
                                    # assign historical flow data
                                    sample_f[r_start_local:r_end_local, c_start_local:c_end_local, :] = f_data[t_now,
                                                                                                        r_start:r_end,
                                                                                                        c_start:c_end,
                                                                                                        :]

                                    # define the matrix to hold the historical transition inputs of AoI
                                    sample_t = np.zeros((block_full_len, block_full_len, 2), dtype=np.float32)
                                    """ this part is a little abstract, the point is to sample the in and out transition
                                        whose duration is less than 2 time intervals """
                                    sample_t[r_start_local:r_end_local, c_start_local:c_end_local, 0] += \
                                        t_data[0, t_now, r_start:r_end, c_start:c_end, r, c]
                                    sample_t[r_start_local:r_end_local, c_start_local:c_end_local, 0] += \
                                        t_data[1, t_now, r_start:r_end, c_start:c_end, r, c]
                                    sample_t[r_start_local:r_end_local, c_start_local:c_end_local, 1] += \
                                        t_data[0, t_now, r, c, r_start:r_end, c_start:c_end]
                                    sample_t[r_start_local:r_end_local, c_start_local:c_end_local, 1] += \
                                        t_data[1, t_now, r, c, r_start:r_end, c_start:c_end]

                                inp_ft_sample.append(np.concatenate([sample_f, sample_t], axis=-1))
                                inp_ex_sample.append(ex_data[t_now, :])

                        """ sampling of inputs of current day, the details are similar to those mentioned above """
                        for int_cnt in range(n_curr_int):
                            t_now = int(t - (n_curr_int - int_cnt))

                            if not local_block_len:
                                sample_f = f_data[t_now, ..., :]

                                sample_t = np.zeros((data_shape[1], data_shape[2], 2), dtype=np.float32)

                                sample_t[..., 0] += t_data[0, t_now, ..., r, c]
                                sample_t[..., 0] += t_data[1, t_now, ..., r, c]
                                sample_t[..., 1] += t_data[0, t_now, r, c, ...]
                                sample_t[..., 1] += t_data[1, t_now, r, c, ...]

                            else:
                                sample_f = np.zeros((block_full_len, block_full_len, 2), dtype=np.float32)
                                sample_f[r_start_local:r_end_local, c_start_local:c_end_local, :] = f_data[
                                                                                                    t_now,
                                                                                                    r_start:r_end,
                                                                                                    c_start:c_end,
                                                                                                    :]

                                sample_t = np.zeros((block_full_len, block_full_len, 2), dtype=np.float32)
                                sample_t[r_start_local:r_end_local, c_start_local:c_end_local, 0] += \
                                    t_data[0, t_now, r_start:r_end, c_start:c_end, r, c]
                                sample_t[r_start_local:r_end_local, c_start_local:c_end_local, 0] += \
                                    t_data[1, t_now, r_start:r_end, c_start:c_end, r, c]
                                sample_t[r_start_local:r_end_local, c_start_local:c_end_local, 1] += \
                                    t_data[0, t_now, r, c, r_start:r_end, c_start:c_end]
                                sample_t[r_start_local:r_end_local, c_start_local:c_end_local, 1] += \
                                    t_data[1, t_now, r, c, r_start:r_end, c_start:c_end]

                            inp_ft_sample.append(np.concatenate([sample_f, sample_t], axis=-1))
                            inp_ex_sample.append(ex_data[t_now, :])

                        """ append the samples of each node to the overall inputs arrays """
                        inp_ft.append(inp_ft_sample)
                        inp_ex.append(inp_ex_sample)

                        if not local_block_len:
                            dec_inp_t_sample = np.zeros((n_pred, data_shape[1], data_shape[2], 2), dtype=np.float32)

                            dec_inp_t_sample[..., 0] += t_data[0, t - 1: t + n_pred - 1, ..., r, c]
                            dec_inp_t_sample[..., 0] += t_data[1, t - 1: t + n_pred - 1, ..., r, c]
                            dec_inp_t_sample[..., 1] += t_data[0, t - 1: t + n_pred - 1, r, c, ...]
                            dec_inp_t_sample[..., 1] += t_data[1, t - 1: t + n_pred - 1, r, c, ...]
                        else:
                            dec_inp_t_sample = np.zeros((n_pred, block_full_len, block_full_len, 2), dtype=np.float32)
                            dec_inp_t_sample[:, r_start_local:r_end_local, c_start_local:c_end_local, 0] += \
                                t_data[0, t - 1: t + n_pred - 1, r_start:r_end, c_start:c_end, r, c]
                            dec_inp_t_sample[:, r_start_local:r_end_local, c_start_local:c_end_local, 0] += \
                                t_data[1, t - 1: t + n_pred - 1, r_start:r_end, c_start:c_end, r, c]
                            dec_inp_t_sample[:, r_start_local:r_end_local, c_start_local:c_end_local, 1] += \
                                t_data[0, t - 1: t + n_pred - 1, r, c, r_start:r_end, c_start:c_end]
                            dec_inp_t_sample[:, r_start_local:r_end_local, c_start_local:c_end_local, 1] += \
                                t_data[1, t - 1: t + n_pred - 1, r, c, r_start:r_end, c_start:c_end]

                        dec_inp_f.append(f_data[t - 1: t + n_pred - 1, r, c, :])
                        dec_inp_t.append(dec_inp_t_sample)

                        dec_inp_ex.append(ex_data[t - 1: t + n_pred - 1, :])

                        cors.append(self.cor_gen.get(r, c))

                        """ generating the ground truth for each sample """
                        y.append(f_data[t: t + n_pred, r, c, :])

                        if not local_block_len:
                            tar_t = np.zeros((n_pred, data_shape[1], data_shape[2], 2), dtype=np.float32)

                            tar_t[..., 0] += t_data[0, t: t + n_pred, ..., r, c]
                            tar_t[..., 0] += t_data[1, t: t + n_pred, ..., r, c]
                            tar_t[..., 1] += t_data[0, t: t + n_pred, r, c, ...]
                            tar_t[..., 1] += t_data[1, t: t + n_pred, r, c, ...]
                        else:
                            tar_t = np.zeros((n_pred, block_full_len, block_full_len, 2), dtype=np.float32)
                            tar_t[:, r_start_local:r_end_local, c_start_local:c_end_local, 0] += \
                                t_data[0, t - 1: t + n_pred - 1, r_start:r_end, c_start:c_end, r, c]
                            tar_t[:, r_start_local:r_end_local, c_start_local:c_end_local, 0] += \
                                t_data[1, t - 1: t + n_pred - 1, r_start:r_end, c_start:c_end, r, c]
                            tar_t[:, r_start_local:r_end_local, c_start_local:c_end_local, 1] += \
                                t_data[0, t - 1: t + n_pred - 1, r, c, r_start:r_end, c_start:c_end]
                            tar_t[:, r_start_local:r_end_local, c_start_local:c_end_local, 1] += \
                                t_data[1, t - 1: t + n_pred - 1, r, c, r_start:r_end, c_start:c_end]

                        y_t.append(tar_t)

                if self.test_model and t + 1 - time_start >= self.test_model:
                    break

            """ convert the inputs arrays to matrices """
            inp_ft = np.array(inp_ft, dtype=np.float32)
            inp_ex = np.array(inp_ex, dtype=np.float32)

            dec_inp_f = np.array(dec_inp_f, dtype=np.float32)
            dec_inp_t = np.array(dec_inp_t, dtype=np.float32)
            dec_inp_ex = np.array(dec_inp_ex, dtype=np.float32)

            cors = np.array(cors, dtype=np.float32)

            y = np.array(y, dtype=np.float32)
            y_t = np.array(y_t, dtype=np.float32)

            """ save the matrices """
            if not self.test_model and not no_save:
                print('Saving .npz files...')
                np.savez_compressed("data/inp_ft_{}_{}.npz".format(self.dataset, datatype), data=inp_ft)
                np.savez_compressed("data/inp_ex_{}_{}.npz".format(self.dataset, datatype), data=inp_ex)
                np.savez_compressed("data/dec_inp_f_{}_{}.npz".format(self.dataset, datatype), data=dec_inp_f)
                np.savez_compressed("data/dec_inp_t_{}_{}.npz".format(self.dataset, datatype), data=dec_inp_t)
                np.savez_compressed("data/dec_inp_ex_{}_{}.npz".format(self.dataset, datatype), data=dec_inp_ex)
                np.savez_compressed("data/cors_{}_{}.npz".format(self.dataset, datatype), data=cors)
                np.savez_compressed("data/y_t_{}_{}.npz".format(self.dataset, datatype), data=y_t)
                np.savez_compressed("data/y_{}_{}.npz".format(self.dataset, datatype), data=y)

            return inp_ft, inp_ex, dec_inp_f, dec_inp_t, dec_inp_ex, cors, y_t, y


if __name__ == "__main__":
    dl = DataLoader(64, test_model=100)
    inp_ft, inp_ex, dec_inp_f, dec_inp_t, dec_inp_ex, cors, y_t, y = dl.generate_data()
