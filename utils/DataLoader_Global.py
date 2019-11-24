import numpy as np
import parameters_nyctaxi as param_taxi
import parameters_nycbike as param_bike
from utils.CordinateGenerator import CordinateGenerator


class DataLoader_Global:
    def __init__(self, d_model, dataset='taxi', test_model=False):
        assert dataset == 'taxi' or 'bike'
        self.dataset = dataset
        self.pmt = param_taxi if dataset == 'taxi' else param_bike
        self.cor_gen = CordinateGenerator(self.pmt.len_r, self.pmt.len_c, d_model)
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
                      load_saved_data=False):  # loading the previous saved data

        assert datatype == 'train' or datatype == 'test'

        """ loading saved data """
        if load_saved_data:
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

            # n_curr_int += 1  # we add one more interval to be taken as the current input

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

                        """ start the samplings of previous weeks """
                        for week_cnt in range(n_hist_week):
                            s_time_w = int(t - (n_hist_week - week_cnt) * 7 * self.pmt.n_int_day - n_int_before)

                            for int_cnt in range(n_hist_int):
                                t_now = s_time_w + int_cnt

                                global_f = f_data[t_now, ...]

                                global_t = np.zeros((data_shape[1], data_shape[2], 4), dtype=np.float32)

                                global_t[..., 0] = t_data[0, t_now, ..., r, c]
                                global_t[..., 1] = t_data[1, t_now, ..., r, c]
                                global_t[..., 2] = t_data[0, t_now, r, c, ...]
                                global_t[..., 3] = t_data[1, t_now, r, c, ...]

                                inp_ft_sample.append(np.concatenate([global_f, global_t], axis=-1))
                                inp_ex_sample.append(ex_data[t_now, :])

                        """ start the samplings of previous days"""
                        for hist_day_cnt in range(n_hist_day):
                            """ define the start time in previous days """
                            s_time_d = int(t - (n_hist_day - hist_day_cnt) * self.pmt.n_int_day - n_int_before)

                            """ generate samples from the previous days """
                            for int_cnt in range(n_hist_int):
                                t_now = s_time_d + int_cnt

                                global_f = f_data[t_now, ...]

                                global_t = np.zeros((data_shape[1], data_shape[2], 4), dtype=np.float32)

                                global_t[..., 0] = t_data[0, t_now, ..., r, c]
                                global_t[..., 1] = t_data[1, t_now, ..., r, c]
                                global_t[..., 2] = t_data[0, t_now, r, c, ...]
                                global_t[..., 3] = t_data[1, t_now, r, c, ...]

                                inp_ft_sample.append(np.concatenate([global_f, global_t], axis=-1))
                                inp_ex_sample.append(ex_data[t_now, :])

                        """ sampling of inputs of current day, the details are similar to those mentioned above """
                        for int_cnt in range(n_curr_int):
                            t_now = int(t - (n_curr_int - int_cnt))

                            global_f = f_data[t_now, ..., :]

                            global_t = np.zeros((data_shape[1], data_shape[2], 4), dtype=np.float32)

                            global_t[..., 0] = t_data[0, t_now, ..., r, c]
                            global_t[..., 1] = t_data[1, t_now, ..., r, c]
                            global_t[..., 2] = t_data[0, t_now, r, c, ...]
                            global_t[..., 3] = t_data[1, t_now, r, c, ...]

                            inp_ft_sample.append(np.concatenate([global_f, global_t], axis=-1))
                            inp_ex_sample.append(ex_data[t_now, :])

                        """ append the samples of each node to the overall inputs arrays """
                        inp_ft.append(inp_ft_sample)
                        inp_ex.append(inp_ex_sample)

                        dec_inp_t_sample = np.zeros((n_pred, data_shape[1], data_shape[2], 4), dtype=np.float32)

                        dec_inp_t_sample[..., 0] = t_data[0, t - 1: t + n_pred - 1, ..., r, c]
                        dec_inp_t_sample[..., 1] = t_data[1, t - 1: t + n_pred - 1, ..., r, c]
                        dec_inp_t_sample[..., 2] = t_data[0, t - 1: t + n_pred - 1, r, c, ...]
                        dec_inp_t_sample[..., 3] = t_data[1, t - 1: t + n_pred - 1, r, c, ...]

                        dec_inp_f.append(f_data[t - 1: t + n_pred - 1, r, c, :])
                        dec_inp_t.append(dec_inp_t_sample)

                        dec_inp_ex.append(ex_data[t - 1: t + n_pred - 1, :])

                        cors.append(self.cor_gen.get(r, c))

                        """ generating the ground truth for each sample """
                        y.append(f_data[t: t + n_pred, r, c, :])

                        tar_t = np.zeros((n_pred, data_shape[1], data_shape[2], 4), dtype=np.float32)

                        tar_t[..., 0] = t_data[0, t: t + n_pred, ..., r, c]
                        tar_t[..., 1] = t_data[1, t: t + n_pred, ..., r, c]
                        tar_t[..., 2] = t_data[0, t: t + n_pred, r, c, ...]
                        tar_t[..., 3] = t_data[1, t: t + n_pred, r, c, ...]

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
    dl = DataLoader_Global(64, test_model=True)
    inp_ft, inp_ex, dec_inp_f, dec_inp_t, dec_inp_ex, cors, y_t, y = dl.generate_data(datatype='train',
                                                                                      load_saved_data=False)
