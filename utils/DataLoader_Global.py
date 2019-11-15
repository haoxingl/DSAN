import numpy as np
import parameters_nyctaxi as param_taxi
import parameters_nycbike as param_bike
from CordinateGenerator import CordinateGenerator


class DataLoader_Global:
    def __init__(self, dataset='taxi'):
        self.dataset = dataset
        if self.dataset == 'taxi':
            self.parameters = param_taxi
        elif self.dataset == 'bike':
            self.parameters = param_bike
        else:
            print('Dataset should be \'taxi\' or \'bike\'')
            raise Exception
        self.cor_gen = CordinateGenerator(self.parameters.len_x, self.parameters.len_y)

    def load_flow(self, datatype='train'):
        if datatype == 'train':
            self.f_train = np.array(np.load(self.parameters.f_train)['flow'], dtype=np.float32) / self.parameters.f_train_max
        else:
            self.f_test = np.array(np.load(self.parameters.f_test)['flow'], dtype=np.float32) / self.parameters.f_train_max

    def load_trans(self, datatype='train'):
        if datatype == 'train':
            self.t_train = np.array(np.load(self.parameters.t_train)['trans'], dtype=np.float32) / self.parameters.t_train_max
        else:
            self.t_test = np.array(np.load(self.parameters.t_test)['trans'], dtype=np.float32) / self.parameters.t_train_max

    """ external_knowledge contains the time and weather information of each time interval """

    def load_external_knowledge(self, datatype='train'):
        if datatype == 'train':
            self.ex_train = np.load(self.parameters.ex_train)['external_knowledge']
        else:
            self.ex_test = np.load(self.parameters.ex_test)['external_knowledge']

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
            curr_inputs_f = np.load("data/curr_inputs_f_{}_{}.npz".format(self.dataset, datatype))['data']
            curr_inputs_t = np.load("data/curr_inputs_t_{}_{}.npz".format(self.dataset, datatype))['data']
            curr_inputs_ex = np.load("data/curr_inputs_ex_{}_{}.npz".format(self.dataset, datatype))['data']
            hist_inputs_f = np.load("data/hist_inputs_f_{}_{}.npz".format(self.dataset, datatype))['data']
            hist_inputs_t = np.load("data/hist_inputs_t_{}_{}.npz".format(self.dataset, datatype))['data']
            hist_inputs_ex = np.load("data/hist_inputs_ex_{}_{}.npz".format(self.dataset, datatype))['data']
            cors = np.load("data/cors_{}_{}.npz".format(self.dataset, datatype))['data']
            dec_inp_f = np.load("data/dec_inp_f_{}_{}.npz".format(self.dataset, datatype))['data']
            dec_inp_t = np.load("data/dec_inp_t_{}_{}.npz".format(self.dataset, datatype))['data']
            y = np.load("data/y_{}_{}.npz".format(self.dataset, datatype))['data']
            y_t = np.load("data/y_t_{}_{}.npz".format(self.dataset, datatype))['data']

            return hist_inputs_f, hist_inputs_t, hist_inputs_ex, curr_inputs_f, curr_inputs_t, curr_inputs_ex, cors, dec_inp_f, dec_inp_t, y_t, y
        else:
            print("Loading {} data...".format(datatype))
            """ loading data """
            self.load_flow(datatype)
            self.load_trans(datatype)
            self.load_external_knowledge(datatype)

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

            n_curr_int += 1  # we add one more interval to be taken as the current input

            """ initialize the array to hold the final inputs """
            y = []  # ground truth of the inflow and outflow of each node at each time interval
            y_t = []  # ground truth of the transitions between each node and its neighbors in the area of interest

            dec_inp_f = []
            dec_inp_t = []

            hist_inputs_f = []  # historical flow inputs from area of interest
            hist_inputs_t = []  # historical transition inputs from area of interest
            hist_inputs_ex = []  # historical external knowledge inputs

            curr_inputs_f = []  # flow inputs of current day
            curr_inputs_t = []  # transition inputs of current day
            curr_inputs_ex = []  # external knowledge inputs of current day

            cors = []

            assert n_hist_week >= 0 and n_hist_day >= 1
            """ set the start time interval to sample the data"""
            s1 = n_hist_day * self.parameters.n_int_day + n_int_before
            s2 = n_hist_week * 7 * self.parameters.n_int_day + n_int_before
            time_start = max(s1, s2)
            time_end = f_data.shape[0] - n_pred

            data_shape = f_data.shape

            for t in range(time_start, time_end):
                if t % 100 == 0:
                    print("Currently at {} interval...".format(t))

                for i in range(f_data.shape[1]):
                    for j in range(f_data.shape[2]):

                        """ initialize the array to hold the samples of each node at each time interval """
                        hist_inputs_f_sample = []
                        hist_inputs_t_sample = []
                        hist_inputs_ex_sample = []

                        curr_inputs_f_sample = []
                        curr_inputs_t_sample = []
                        curr_inputs_ex_sample = []

                        """ start the samplings of previous weeks """
                        for week_cnt in range(n_hist_week):
                            this_week_start_time = int(t - (
                                    n_hist_week - week_cnt) * 7 * self.parameters.n_int_day - n_int_before)

                            for int_cnt in range(n_hist_int):
                                t_now = this_week_start_time + int_cnt

                                local_flow = f_data[t_now, :, :, :]

                                local_trans = np.zeros((data_shape[1], data_shape[2], 4), dtype=np.float32)

                                local_trans[..., 0] = t_data[0, t_now, ..., i, j]
                                local_trans[..., 1] = t_data[1, t_now, ..., i, j]
                                local_trans[..., 2] = t_data[0, t_now, i, j, ...]
                                local_trans[..., 3] = t_data[1, t_now, i, j, ...]

                                hist_inputs_f_sample.append(local_flow)
                                hist_inputs_t_sample.append(local_trans)
                                hist_inputs_ex_sample.append(ex_data[t_now, :])

                        """ start the samplings of previous days"""
                        for hist_day_cnt in range(n_hist_day):
                            """ define the start time in previous days """
                            hist_day_start_time = int(t - (
                                    n_hist_day - hist_day_cnt) * self.parameters.n_int_day - n_int_before)

                            """ generate samples from the previous days """
                            for int_cnt in range(n_hist_int):
                                t_now = hist_day_start_time + int_cnt

                                local_flow = f_data[t_now, ...]

                                local_trans = np.zeros((data_shape[1], data_shape[2], 4), dtype=np.float32)

                                local_trans[..., 0] = t_data[0, t_now, ..., i, j]
                                local_trans[..., 1] = t_data[1, t_now, ..., i, j]
                                local_trans[..., 2] = t_data[0, t_now, i, j, ...]
                                local_trans[..., 3] = t_data[1, t_now, i, j, ...]

                                hist_inputs_f_sample.append(local_flow)
                                hist_inputs_t_sample.append(local_trans)
                                hist_inputs_ex_sample.append(ex_data[t_now, :])

                        """ sampling of inputs of current day, the details are similar to those mentioned above """
                        for int_cnt in range(n_curr_int):
                            t_now = int(t - (n_curr_int - int_cnt))

                            local_flow = f_data[t_now, ..., :]

                            local_trans = np.zeros((data_shape[1], data_shape[2], 4), dtype=np.float32)

                            local_trans[..., 0] = t_data[0, t_now, ..., i, j]
                            local_trans[..., 1] = t_data[1, t_now, ..., i, j]
                            local_trans[..., 2] = t_data[0, t_now, i, j, ...]
                            local_trans[..., 3] = t_data[1, t_now, i, j, ...]

                            curr_inputs_f_sample.append(local_flow)
                            curr_inputs_t_sample.append(local_trans)
                            curr_inputs_ex_sample.append(ex_data[t_now, :])

                        """ append the samples of each node to the overall inputs arrays """
                        curr_inputs_f.append(np.array(curr_inputs_f_sample, dtype=np.float32))
                        curr_inputs_t.append(np.array(curr_inputs_t_sample, dtype=np.float32))
                        curr_inputs_ex.append(np.array(curr_inputs_ex_sample, dtype=np.float32))
                        hist_inputs_f.append(np.array(hist_inputs_f_sample, dtype=np.float32))
                        hist_inputs_t.append(np.array(hist_inputs_t_sample, dtype=np.float32))
                        hist_inputs_ex.append(np.array(hist_inputs_ex_sample, dtype=np.float32))

                        cors.append(self.cor_gen.get(j, i))

                        dec_inp_f.append(f_data[t - 1 : t + n_pred - 1, i, j, :])

                        """ generating the ground truth for each sample """
                        y.append(f_data[t : t + n_pred, i, j, :])

                        dec_inp_t_sample = np.zeros((n_pred, data_shape[1], data_shape[2], 4), dtype=np.float32)

                        dec_inp_t_sample[..., 0] = t_data[0, t - 1 : t + n_pred - 1, ..., i, j]
                        dec_inp_t_sample[..., 1] = t_data[1, t - 1 : t + n_pred - 1, ..., i, j]
                        dec_inp_t_sample[..., 2] = t_data[0, t - 1 : t + n_pred - 1, i, j, ...]
                        dec_inp_t_sample[..., 3] = t_data[1, t - 1 : t + n_pred - 1, i, j, ...]

                        dec_inp_t.append(dec_inp_t_sample)

                        tar_t = np.zeros((n_pred, data_shape[1], data_shape[2], 4), dtype=np.float32)

                        tar_t[..., 0] = t_data[0, t : t + n_pred, ..., i, j]
                        tar_t[..., 1] = t_data[1, t : t + n_pred, ..., i, j]
                        tar_t[..., 2] = t_data[0, t : t + n_pred, i, j, ...]
                        tar_t[..., 3] = t_data[1, t : t + n_pred, i, j, ...]

                        y_t.append(tar_t)

            """ convert the inputs arrays to matrices """
            curr_inputs_f = np.array(curr_inputs_f, dtype=np.float32).transpose((0, 2, 3, 1, 4))
            curr_inputs_t = np.array(curr_inputs_t, dtype=np.float32).transpose((0, 2, 3, 1, 4))
            curr_inputs_ex = np.array(curr_inputs_ex, dtype=np.float32)
            hist_inputs_f = np.array(hist_inputs_f, dtype=np.float32).transpose((0, 2, 3, 1, 4))
            hist_inputs_t = np.array(hist_inputs_t, dtype=np.float32).transpose((0, 2, 3, 1, 4))
            hist_inputs_ex = np.array(hist_inputs_ex, dtype=np.float32)

            cors = np.array(cors, dtype=np.int64)
            dec_inp_f = np.array(dec_inp_f, dtype=np.float32)
            dec_inp_t = np.array(dec_inp_t, dtype=np.float32)
            y = np.array(y, dtype=np.float32)
            y_t = np.array(y_t, dtype=np.float32)

            """ save the matrices """
            np.savez_compressed("data/curr_inputs_f_{}_{}.npz".format(self.dataset, datatype), data=curr_inputs_f)
            np.savez_compressed("data/curr_inputs_t_{}_{}.npz".format(self.dataset, datatype), data=curr_inputs_t)
            np.savez_compressed("data/curr_inputs_ex_{}_{}.npz".format(self.dataset, datatype), data=curr_inputs_ex)
            np.savez_compressed("data/hist_inputs_f_{}_{}.npz".format(self.dataset, datatype), data=hist_inputs_f)
            np.savez_compressed("data/hist_inputs_t_{}_{}.npz".format(self.dataset, datatype), data=hist_inputs_t)
            np.savez_compressed("data/hist_inputs_ex_{}_{}.npz".format(self.dataset, datatype), data=hist_inputs_ex)
            np.savez_compressed("data/cors_{}_{}.npz".format(self.dataset, datatype), data=cors)
            np.savez_compressed("data/dec_inp_f_{}_{}.npz".format(self.dataset, datatype), data=dec_inp_f)
            np.savez_compressed("data/dec_inp_t_{}_{}.npz".format(self.dataset, datatype), data=dec_inp_t)
            np.savez_compressed("data/y_{}_{}.npz".format(self.dataset, datatype), data=y)
            np.savez_compressed("data/y_t_{}_{}.npz".format(self.dataset, datatype), data=y_t)

            return hist_inputs_f, hist_inputs_t, hist_inputs_ex, curr_inputs_f, curr_inputs_t, curr_inputs_ex, cors, dec_inp_f, dec_inp_t, y_t, y

if __name__ == "__main__":
    dl = DataLoader_Global()
    hist_inputs_f, hist_inputs_t, hist_inputs_ex, curr_inputs_f, curr_inputs_t, curr_inputs_ex, cors, dec_inp_f, dec_inp_t, y_t, y = \
        dl.generate_data(load_saved_data=True)
