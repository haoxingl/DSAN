import numpy as np
from utils.CordinateGenerator import CordinateGenerator

import parameters_nyctaxi as param_taxi
import parameters_nycbike as param_bike
import parameters_ctm as param_ctm


class DataLoader:
    def __init__(self, d_model, dataset='taxi', local_block_len=3, local_block_len_g=5, test_model=False):
        assert dataset in ['taxi', 'bike', 'ctm']
        self.dataset = dataset
        pmt = None
        pmt = param_taxi if dataset == 'taxi' else pmt
        pmt = param_bike if dataset == 'bike' else pmt
        pmt = param_ctm if dataset == 'ctm' else pmt
        self.pmt = pmt
        self.local_block_len = local_block_len
        self.local_block_len_g = local_block_len_g
        self.cor_gen = CordinateGenerator(self.pmt.len_r, self.pmt.len_c, d_model, local_block_len=local_block_len)
        self.cor_gen_g = CordinateGenerator(self.pmt.len_r, self.pmt.len_c, d_model, local_block_len=local_block_len_g)
        self.test_model = test_model

    def load_data(self, datatype='train'):
        if datatype == 'train':
            data = np.load(self.pmt.data_train)
        elif datatype == 'val':
            data = np.load(self.pmt.data_val)
        else:
            data = np.load(self.pmt.data_test)

        if self.dataset in ['taxi', 'bike']:
            self.data_mtx = np.array(data['flow'], dtype=np.float32) / self.pmt.data_max
            self.t_mtx = np.array(data['trans'], dtype=np.float32) / self.pmt.t_max
            self.ex_mtx = data['ex_knlg']
        else:
            self.data_mtx = np.array(data['data'], dtype=np.float32)
            self.data_mtx[..., 0] = self.data_mtx[..., 0] / self.pmt.data_max[0]
            self.data_mtx[..., 1] = self.data_mtx[..., 1] / self.pmt.data_max[1]
            self.ex_mtx = data['ex_knlg']

    def generate_data(self, datatype='train',
                      n_hist_week=1,  # number previous weeks we generate the sample from.
                      n_hist_day=3,  # number of the previous days we generate the sample from
                      n_hist_int=1,  # number of intervals we sample in the previous weeks, days
                      n_curr_int=1,  # number of intervals we sample in the current day
                      n_int_before=0,  # number of intervals before the predicted interval
                      n_pred=12,
                      st_revert=False,
                      no_save=False,
                      load_saved_data=False):  # loading the previous saved data

        assert datatype in ['train', 'val', 'test']

        """ loading saved data """
        if load_saved_data and not self.test_model:
            print('Loading {} data from .npzs...'.format(datatype))
            inp_g = np.load("data/inp_g_{}_{}.npz".format(self.dataset, datatype))['data']
            inp_l = np.load("data/inp_l_{}_{}.npz".format(self.dataset, datatype))['data']
            inp_ex = np.load("data/inp_ex_{}_{}.npz".format(self.dataset, datatype))['data']
            dec_inp = np.load("data/dec_inp_{}_{}.npz".format(self.dataset, datatype))['data']
            dec_inp_ex = np.load("data/dec_inp_ex_{}_{}.npz".format(self.dataset, datatype))['data']
            cors = np.load("data/cors_{}_{}.npz".format(self.dataset, datatype))['data']
            cors_g = np.load("data/cors_g_{}_{}.npz".format(self.dataset, datatype))['data']
            y = np.load("data/y_{}_{}.npz".format(self.dataset, datatype))['data']

            return inp_g, inp_l, inp_ex, dec_inp, dec_inp_ex, cors, cors_g, y
        else:
            local_block_len = self.local_block_len
            local_block_len_g = self.local_block_len_g

            print("Loading {} data...".format(datatype))
            """ loading data """
            self.load_data(datatype)

            data_mtx = self.data_mtx
            ex_mtx = self.ex_mtx
            data_shape = data_mtx.shape
            no_ctm = self.dataset in ['taxi', 'bike']
            if no_ctm:
                t_mtx = self.t_mtx

            if local_block_len:
                block_full_len = 2 * local_block_len + 1

            if local_block_len_g:
                block_full_len_g = 2 * local_block_len_g + 1

            """ initialize the array to hold the final inputs """

            inp_g = []
            inp_l = []
            inp_ex = []

            cors_g = []
            cors = []

            dec_inp = []
            dec_inp_ex = []

            y = []

            assert n_hist_week >= 0 and n_hist_day >= 0 and n_hist_day < 7
            """ set the start time interval to sample the data"""
            s1 = n_hist_day * self.pmt.n_int_day + n_int_before
            s2 = n_hist_week * 7 * self.pmt.n_int_day + n_int_before
            time_start = max(s1, s2)
            time_end = data_shape[0] - n_pred

            for t in range(time_start, time_end):
                if (t - time_start) % 100 == 0:
                    print("Loading {}/{}".format(t - time_start, time_end - time_start))

                for r in range(data_shape[1]):
                    for c in range(data_shape[2]):

                        """ initialize the array to hold the samples of each node at each time interval """

                        inp_g_sample = []
                        inp_l_sample = []
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

                        if local_block_len_g:
                            """ initialize the boundaries of the area of interest """
                            r_start_g = r - local_block_len_g  # the start location of each AoI
                            c_start_g = c - local_block_len_g

                            """ adjust the start location if it is on the boundaries of the grid map """
                            if r_start_g < 0:
                                r_start_local_g = 0 - r_start_g
                                r_start_g = 0
                            else:
                                r_start_local_g = 0
                            if c_start_g < 0:
                                c_start_local_g = 0 - c_start_g
                                c_start_g = 0
                            else:
                                c_start_local_g = 0

                            r_end_g = r + local_block_len_g + 1  # the end location of each AoI
                            c_end_g = c + local_block_len_g + 1
                            if r_end_g >= data_shape[1]:
                                r_end_local_g = block_full_len_g - (r_end_g - data_shape[1])
                                r_end_g = data_shape[1]
                            else:
                                r_end_local_g = block_full_len_g
                            if c_end_g >= data_shape[2]:
                                c_end_local_g = block_full_len_g - (c_end_g - data_shape[2])
                                c_end_g = data_shape[2]
                            else:
                                c_end_local_g = block_full_len_g

                        """ start the samplings of previous weeks """
                        for week_cnt in range(n_hist_week):
                            s_time_w = int(t - (n_hist_week - week_cnt) * 7 * self.pmt.n_int_day - n_int_before)

                            for int_cnt in range(n_hist_int):
                                t_now = s_time_w + int_cnt

                                if not local_block_len_g:
                                    one_inp_g = data_mtx[t_now, ...]

                                    if no_ctm:
                                        one_inp_g_t = np.zeros((data_shape[1], data_shape[2], 2), dtype=np.float32)

                                        one_inp_g_t[..., 0] += t_mtx[0, t_now, ..., r, c]
                                        one_inp_g_t[..., 0] += t_mtx[1, t_now, ..., r, c]
                                        one_inp_g_t[..., 1] += t_mtx[0, t_now, r, c, ...]
                                        one_inp_g_t[..., 1] += t_mtx[1, t_now, r, c, ...]

                                else:
                                    one_inp_g = np.zeros((block_full_len_g, block_full_len_g, 2), dtype=np.float32)
                                    one_inp_g[r_start_local_g:r_end_local_g, c_start_local_g:c_end_local_g,
                                    :] = data_mtx[
                                         t_now,
                                         r_start_g:r_end_g,
                                         c_start_g:c_end_g,
                                         :]

                                    if no_ctm:
                                        one_inp_g_t = np.zeros((block_full_len_g, block_full_len_g, 2),
                                                               dtype=np.float32)
                                        one_inp_g_t[r_start_local_g:r_end_local_g, c_start_local_g:c_end_local_g,
                                        0] += \
                                            t_mtx[0, t_now, r_start_g:r_end_g, c_start_g:c_end_g, r, c]
                                        one_inp_g_t[r_start_local_g:r_end_local_g, c_start_local_g:c_end_local_g,
                                        0] += \
                                            t_mtx[1, t_now, r_start_g:r_end_g, c_start_g:c_end_g, r, c]
                                        one_inp_g_t[r_start_local_g:r_end_local_g, c_start_local_g:c_end_local_g,
                                        1] += \
                                            t_mtx[0, t_now, r, c, r_start_g:r_end_g, c_start_g:c_end_g]
                                        one_inp_g_t[r_start_local_g:r_end_local_g, c_start_local_g:c_end_local_g,
                                        1] += \
                                            t_mtx[1, t_now, r, c, r_start_g:r_end_g, c_start_g:c_end_g]

                                inp_g_sample.append(
                                    np.concatenate([one_inp_g, one_inp_g_t], axis=-1) if no_ctm else one_inp_g)

                                if not local_block_len:
                                    one_inp_l = data_mtx[t_now, ...]

                                    if no_ctm:
                                        one_inp_l_t = np.zeros((data_shape[1], data_shape[2], 2), dtype=np.float32)

                                        one_inp_l_t[..., 0] += t_mtx[0, t_now, ..., r, c]
                                        one_inp_l_t[..., 0] += t_mtx[1, t_now, ..., r, c]
                                        one_inp_l_t[..., 1] += t_mtx[0, t_now, r, c, ...]
                                        one_inp_l_t[..., 1] += t_mtx[1, t_now, r, c, ...]

                                else:
                                    one_inp_l = np.zeros((block_full_len, block_full_len, 2), dtype=np.float32)
                                    one_inp_l[r_start_local:r_end_local, c_start_local:c_end_local, :] = data_mtx[
                                                                                                       t_now,
                                                                                                       r_start:r_end,
                                                                                                       c_start:c_end,
                                                                                                       :]

                                    if no_ctm:
                                        one_inp_l_t = np.zeros((block_full_len, block_full_len, 2), dtype=np.float32)
                                        one_inp_l_t[r_start_local:r_end_local, c_start_local:c_end_local, 0] += \
                                            t_mtx[0, t_now, r_start:r_end, c_start:c_end, r, c]
                                        one_inp_l_t[r_start_local:r_end_local, c_start_local:c_end_local, 0] += \
                                            t_mtx[1, t_now, r_start:r_end, c_start:c_end, r, c]
                                        one_inp_l_t[r_start_local:r_end_local, c_start_local:c_end_local, 1] += \
                                            t_mtx[0, t_now, r, c, r_start:r_end, c_start:c_end]
                                        one_inp_l_t[r_start_local:r_end_local, c_start_local:c_end_local, 1] += \
                                            t_mtx[1, t_now, r, c, r_start:r_end, c_start:c_end]

                                inp_l_sample.append(np.concatenate([one_inp_l, one_inp_l_t], axis=-1) if no_ctm else one_inp_l)
                                inp_ex_sample.append(ex_mtx[t_now, :])

                        """ start the samplings of previous days"""
                        for hist_day_cnt in range(n_hist_day):
                            """ define the start time in previous days """
                            s_time_d = int(t - (n_hist_day - hist_day_cnt) * self.pmt.n_int_day - n_int_before)

                            """ generate samples from the previous days """
                            for int_cnt in range(n_hist_int):
                                t_now = s_time_d + int_cnt

                                if not local_block_len_g:
                                    one_inp_g = data_mtx[t_now, ...]

                                    if no_ctm:
                                        one_inp_g_t = np.zeros((data_shape[1], data_shape[2], 2), dtype=np.float32)

                                        one_inp_g_t[..., 0] += t_mtx[0, t_now, ..., r, c]
                                        one_inp_g_t[..., 0] += t_mtx[1, t_now, ..., r, c]
                                        one_inp_g_t[..., 1] += t_mtx[0, t_now, r, c, ...]
                                        one_inp_g_t[..., 1] += t_mtx[1, t_now, r, c, ...]

                                else:
                                    one_inp_g = np.zeros((block_full_len_g, block_full_len_g, 2), dtype=np.float32)
                                    one_inp_g[r_start_local_g:r_end_local_g, c_start_local_g:c_end_local_g,
                                    :] = data_mtx[
                                         t_now,
                                         r_start_g:r_end_g,
                                         c_start_g:c_end_g,
                                         :]

                                    if no_ctm:
                                        one_inp_g_t = np.zeros((block_full_len_g, block_full_len_g, 2),
                                                               dtype=np.float32)
                                        one_inp_g_t[r_start_local_g:r_end_local_g, c_start_local_g:c_end_local_g,
                                        0] += \
                                            t_mtx[0, t_now, r_start_g:r_end_g, c_start_g:c_end_g, r, c]
                                        one_inp_g_t[r_start_local_g:r_end_local_g, c_start_local_g:c_end_local_g,
                                        0] += \
                                            t_mtx[1, t_now, r_start_g:r_end_g, c_start_g:c_end_g, r, c]
                                        one_inp_g_t[r_start_local_g:r_end_local_g, c_start_local_g:c_end_local_g,
                                        1] += \
                                            t_mtx[0, t_now, r, c, r_start_g:r_end_g, c_start_g:c_end_g]
                                        one_inp_g_t[r_start_local_g:r_end_local_g, c_start_local_g:c_end_local_g,
                                        1] += \
                                            t_mtx[1, t_now, r, c, r_start_g:r_end_g, c_start_g:c_end_g]

                                inp_g_sample.append(
                                    np.concatenate([one_inp_g, one_inp_g_t], axis=-1) if no_ctm else one_inp_g)

                                if not local_block_len:
                                    one_inp_l = data_mtx[t_now, ...]

                                    if no_ctm:
                                        one_inp_l_t = np.zeros((data_shape[1], data_shape[2], 2), dtype=np.float32)

                                        one_inp_l_t[..., 0] += t_mtx[0, t_now, ..., r, c]
                                        one_inp_l_t[..., 0] += t_mtx[1, t_now, ..., r, c]
                                        one_inp_l_t[..., 1] += t_mtx[0, t_now, r, c, ...]
                                        one_inp_l_t[..., 1] += t_mtx[1, t_now, r, c, ...]

                                else:
                                    one_inp_l = np.zeros((block_full_len, block_full_len, 2), dtype=np.float32)
                                    one_inp_l[r_start_local:r_end_local, c_start_local:c_end_local, :] = data_mtx[
                                                                                                       t_now,
                                                                                                       r_start:r_end,
                                                                                                       c_start:c_end,
                                                                                                       :]

                                    if no_ctm:
                                        one_inp_l_t = np.zeros((block_full_len, block_full_len, 2), dtype=np.float32)
                                        one_inp_l_t[r_start_local:r_end_local, c_start_local:c_end_local, 0] += \
                                            t_mtx[0, t_now, r_start:r_end, c_start:c_end, r, c]
                                        one_inp_l_t[r_start_local:r_end_local, c_start_local:c_end_local, 0] += \
                                            t_mtx[1, t_now, r_start:r_end, c_start:c_end, r, c]
                                        one_inp_l_t[r_start_local:r_end_local, c_start_local:c_end_local, 1] += \
                                            t_mtx[0, t_now, r, c, r_start:r_end, c_start:c_end]
                                        one_inp_l_t[r_start_local:r_end_local, c_start_local:c_end_local, 1] += \
                                            t_mtx[1, t_now, r, c, r_start:r_end, c_start:c_end]

                                inp_l_sample.append(np.concatenate([one_inp_l, one_inp_l_t], axis=-1) if no_ctm else one_inp_l)
                                inp_ex_sample.append(ex_mtx[t_now, :])

                        """ sampling of inputs of current day, the details are similar to those mentioned above """
                        for int_cnt in range(n_curr_int):
                            t_now = int(t - (n_curr_int - int_cnt))

                            if not local_block_len_g:
                                one_inp_g = data_mtx[t_now, ...]

                                if no_ctm:
                                    one_inp_g_t = np.zeros((data_shape[1], data_shape[2], 2), dtype=np.float32)

                                    one_inp_g_t[..., 0] += t_mtx[0, t_now, ..., r, c]
                                    one_inp_g_t[..., 0] += t_mtx[1, t_now, ..., r, c]
                                    one_inp_g_t[..., 1] += t_mtx[0, t_now, r, c, ...]
                                    one_inp_g_t[..., 1] += t_mtx[1, t_now, r, c, ...]

                            else:
                                one_inp_g = np.zeros((block_full_len_g, block_full_len_g, 2), dtype=np.float32)
                                one_inp_g[r_start_local_g:r_end_local_g, c_start_local_g:c_end_local_g,
                                :] = data_mtx[
                                     t_now,
                                     r_start_g:r_end_g,
                                     c_start_g:c_end_g,
                                     :]

                                if no_ctm:
                                    one_inp_g_t = np.zeros((block_full_len_g, block_full_len_g, 2),
                                                           dtype=np.float32)
                                    one_inp_g_t[r_start_local_g:r_end_local_g, c_start_local_g:c_end_local_g,
                                    0] += \
                                        t_mtx[0, t_now, r_start_g:r_end_g, c_start_g:c_end_g, r, c]
                                    one_inp_g_t[r_start_local_g:r_end_local_g, c_start_local_g:c_end_local_g,
                                    0] += \
                                        t_mtx[1, t_now, r_start_g:r_end_g, c_start_g:c_end_g, r, c]
                                    one_inp_g_t[r_start_local_g:r_end_local_g, c_start_local_g:c_end_local_g,
                                    1] += \
                                        t_mtx[0, t_now, r, c, r_start_g:r_end_g, c_start_g:c_end_g]
                                    one_inp_g_t[r_start_local_g:r_end_local_g, c_start_local_g:c_end_local_g,
                                    1] += \
                                        t_mtx[1, t_now, r, c, r_start_g:r_end_g, c_start_g:c_end_g]

                            inp_g_sample.append(
                                np.concatenate([one_inp_g, one_inp_g_t], axis=-1) if no_ctm else one_inp_g)

                            if not local_block_len:
                                one_inp_l = data_mtx[t_now, ...]

                                if no_ctm:
                                    one_inp_l_t = np.zeros((data_shape[1], data_shape[2], 2), dtype=np.float32)

                                    one_inp_l_t[..., 0] += t_mtx[0, t_now, ..., r, c]
                                    one_inp_l_t[..., 0] += t_mtx[1, t_now, ..., r, c]
                                    one_inp_l_t[..., 1] += t_mtx[0, t_now, r, c, ...]
                                    one_inp_l_t[..., 1] += t_mtx[1, t_now, r, c, ...]

                            else:
                                one_inp_l = np.zeros((block_full_len, block_full_len, 2), dtype=np.float32)
                                one_inp_l[r_start_local:r_end_local, c_start_local:c_end_local, :] = data_mtx[
                                                                                                   t_now,
                                                                                                   r_start:r_end,
                                                                                                   c_start:c_end,
                                                                                                   :]

                                if no_ctm:
                                    one_inp_l_t = np.zeros((block_full_len, block_full_len, 2), dtype=np.float32)
                                    one_inp_l_t[r_start_local:r_end_local, c_start_local:c_end_local, 0] += \
                                        t_mtx[0, t_now, r_start:r_end, c_start:c_end, r, c]
                                    one_inp_l_t[r_start_local:r_end_local, c_start_local:c_end_local, 0] += \
                                        t_mtx[1, t_now, r_start:r_end, c_start:c_end, r, c]
                                    one_inp_l_t[r_start_local:r_end_local, c_start_local:c_end_local, 1] += \
                                        t_mtx[0, t_now, r, c, r_start:r_end, c_start:c_end]
                                    one_inp_l_t[r_start_local:r_end_local, c_start_local:c_end_local, 1] += \
                                        t_mtx[1, t_now, r, c, r_start:r_end, c_start:c_end]

                            inp_l_sample.append(np.concatenate([one_inp_l, one_inp_l_t], axis=-1) if no_ctm else one_inp_l)
                            inp_ex_sample.append(ex_mtx[t_now, :])

                        """ append the samples of each node to the overall inputs arrays """
                        inp_g.append(inp_g_sample)
                        inp_l.append(inp_l_sample)
                        inp_ex.append(inp_ex_sample)

                        dec_inp.append(data_mtx[t - 1: t + n_pred - 1, r, c, :])

                        dec_inp_ex.append(ex_mtx[t - 1: t + n_pred - 1, :])

                        cors.append(self.cor_gen.get(r, c))
                        cors_g.append(self.cor_gen_g.get(r, c))

                        """ generating the ground truth for each sample """
                        y.append(data_mtx[t: t + n_pred, r, c, :])

                if self.test_model and t + 1 - time_start >= self.test_model:
                    break

            """ convert the inputs arrays to matrices """
            inp_g = np.array(inp_g, dtype=np.float32)
            inp_l = np.array(inp_l, dtype=np.float32)
            inp_ex = np.array(inp_ex, dtype=np.float32)

            dec_inp = np.array(dec_inp, dtype=np.float32)
            dec_inp_ex = np.array(dec_inp_ex, dtype=np.float32)

            cors = np.array(cors, dtype=np.float32)
            cors_g = np.array(cors_g, dtype=np.float32)

            y = np.array(y, dtype=np.float32)

            if st_revert:
                inp_g = inp_g.transpose((0, 2, 3, 1, 4))
                inp_l = inp_l.transpose((0, 2, 3, 1, 4))

            """ save the matrices """
            if not self.test_model and not no_save:
                print('Saving .npz files...')
                np.savez_compressed("data/inp_g_{}_{}.npz".format(self.dataset, datatype), data=inp_g)
                np.savez_compressed("data/inp_l_{}_{}.npz".format(self.dataset, datatype), data=inp_l)
                np.savez_compressed("data/inp_ex_{}_{}.npz".format(self.dataset, datatype), data=inp_ex)
                np.savez_compressed("data/dec_inp_{}_{}.npz".format(self.dataset, datatype), data=dec_inp)
                np.savez_compressed("data/dec_inp_ex_{}_{}.npz".format(self.dataset, datatype), data=dec_inp_ex)
                np.savez_compressed("data/cors_{}_{}.npz".format(self.dataset, datatype), data=cors)
                np.savez_compressed("data/cors_g_{}_{}.npz".format(self.dataset, datatype), data=cors_g)
                np.savez_compressed("data/y_{}_{}.npz".format(self.dataset, datatype), data=y)

            return inp_g, inp_l, inp_ex, dec_inp, dec_inp_ex, cors, cors_g, y


if __name__ == "__main__":
    dl = DataLoader(64)
    inp_g, inp_l, inp_ex, dec_inp, dec_inp_ex, cors, cors_g, y = dl.generate_data()
    inp_g, inp_l, inp_ex, dec_inp, dec_inp_ex, cors, cors_g, y = dl.generate_data(datatype='val')
    inp_g, inp_l, inp_ex, dec_inp, dec_inp_ex, cors, cors_g, y = dl.generate_data(datatype='test')
    inp_g, inp_l, inp_ex, dec_inp, dec_inp_ex, cors, cors_g, y = dl.generate_data(load_saved_data=True)
    inp_g, inp_l, inp_ex, dec_inp, dec_inp_ex, cors, cors_g, y = dl.generate_data(load_saved_data=True,
                                                                                datatype='val')
    inp_g, inp_l, inp_ex, dec_inp, dec_inp_ex, cors, cors_g, y = dl.generate_data(load_saved_data=True,
                                                                                datatype='test')

    dl = DataLoader(64, dataset='bike')
    inp_g, inp_l, inp_ex, dec_inp, dec_inp_ex, cors, cors_g, y = dl.generate_data()
    inp_g, inp_l, inp_ex, dec_inp, dec_inp_ex, cors, cors_g, y = dl.generate_data(datatype='val')
    inp_g, inp_l, inp_ex, dec_inp, dec_inp_ex, cors, cors_g, y = dl.generate_data(datatype='test')
    inp_g, inp_l, inp_ex, dec_inp, dec_inp_ex, cors, cors_g, y = dl.generate_data(load_saved_data=True)
    inp_g, inp_l, inp_ex, dec_inp, dec_inp_ex, cors, cors_g, y = dl.generate_data(load_saved_data=True,
                                                                                datatype='val')
    inp_g, inp_l, inp_ex, dec_inp, dec_inp_ex, cors, cors_g, y = dl.generate_data(load_saved_data=True,
                                                                                datatype='test')

    dl = DataLoader(64, dataset='ctm')
    inp_g, inp_l, inp_ex, dec_inp, dec_inp_ex, cors, cors_g, y = dl.generate_data()
    inp_g, inp_l, inp_ex, dec_inp, dec_inp_ex, cors, cors_g, y = dl.generate_data(datatype='val')
    inp_g, inp_l, inp_ex, dec_inp, dec_inp_ex, cors, cors_g, y = dl.generate_data(datatype='test')
    inp_g, inp_l, inp_ex, dec_inp, dec_inp_ex, cors, cors_g, y = dl.generate_data(load_saved_data=True)
    inp_g, inp_l, inp_ex, dec_inp, dec_inp_ex, cors, cors_g, y = dl.generate_data(load_saved_data=True,
                                                                                datatype='val')
    inp_g, inp_l, inp_ex, dec_inp, dec_inp_ex, cors, cors_g, y = dl.generate_data(load_saved_data=True,
                                                                                datatype='test')
