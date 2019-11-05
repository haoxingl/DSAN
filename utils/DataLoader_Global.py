import numpy as np
import parameters_nyctaxi as param_taxi
import parameters_nycbike as param_bike


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

    def load_flow(self):
        self.flow_train = np.array(np.load(self.parameters.flow_train)['flow'],
                                   dtype=np.float32) / self.parameters.flow_train_max
        self.flow_test = np.array(np.load(self.parameters.flow_test)['flow'],
                                  dtype=np.float32) / self.parameters.flow_train_max

    def load_trans(self):
        self.trans_train = np.array(np.load(self.parameters.trans_train)['trans'],
                                    dtype=np.float32) / self.parameters.trans_train_max
        self.trans_test = np.array(np.load(self.parameters.trans_test)['trans'],
                                   dtype=np.float32) / self.parameters.trans_train_max

    """ external_knowledge contains the time and weather information of each time interval """

    def load_external_knowledge(self):
        self.ex_knlg_data_train = np.load(self.parameters.external_knowledge_train)['external_knowledge']
        self.ex_knlg_data_test = np.load(self.parameters.external_knowledge_test)['external_knowledge']

    def generate_data(self, datatype='train',
                      num_weeks_hist=0,  # number previous weeks we generate the sample from.
                      num_days_hist=7,  # number of the previous days we generate the sample from
                      num_intervals_hist=3,  # number of intervals we sample in the previous weeks, days
                      num_intervals_curr=1,  # number of intervals we sample in the current day
                      num_intervals_before_predict=1,
                      # number of intervals before the interval to predict in each day, used to adjust the position of the sliding windows
                      local_block_len_half=3,  # half of the length of local convolution block
                      load_saved_data=False):  # loading the previous saved data

        """ loading saved data """
        if load_saved_data:
            print('Loading {} data from .npzs...'.format(datatype))
            flow_inputs_curr = np.load("data/flow_inputs_curr_{}_{}.npz".format(self.dataset, datatype))['data']
            transition_inputs_curr = np.load("data/transition_inputs_curr_{}_{}.npz".format(self.dataset, datatype))[
                'data']
            ex_inputs_curr = np.load("data/ex_inputs_curr_{}_{}.npz".format(self.dataset, datatype))['data']
            flow_inputs_hist = np.load("data/flow_inputs_hist_{}_{}.npz".format(self.dataset, datatype))['data']
            transition_inputs_hist = np.load("data/transition_inputs_hist_{}_{}.npz".format(self.dataset, datatype))[
                'data']
            ex_inputs_hist = np.load("data/ex_inputs_hist_{}_{}.npz".format(self.dataset, datatype))['data']
            xs = np.load("data/xs_{}_{}.npz".format(self.dataset, datatype))['data']
            ys = np.load("data/ys_{}_{}.npz".format(self.dataset, datatype))['data']
            ys_transitions = np.load("data/ys_transitions_{}_{}.npz".format(self.dataset, datatype))['data']

            return flow_inputs_hist, transition_inputs_hist, ex_inputs_hist, flow_inputs_curr, transition_inputs_curr, ex_inputs_curr, xs, ys_transitions, ys
        else:
            print("Loading {} data...".format(datatype))
            """ loading data """
            self.load_flow()
            self.load_trans()
            self.load_external_knowledge()

            if datatype == "train":
                flow_data = self.flow_train
                trans_data = self.trans_train
                ex_knlg_data = self.ex_knlg_data_train
            elif datatype == "test":
                flow_data = self.flow_test
                trans_data = self.trans_test
                ex_knlg_data = self.ex_knlg_data_test
            else:
                print("Please select **train** or **test**")
                raise Exception

            num_intervals_curr += 1  # we add one more interval to be taken as the current input

            """ initialize the array to hold the final inputs """
            ys = []  # ground truth of the inflow and outflow of each node at each time interval
            ys_transitions = []  # ground truth of the transitions between each node and its neighbors in the area of interest

            xs = []

            flow_inputs_hist = []  # historical flow inputs from area of interest
            transition_inputs_hist = []  # historical transition inputs from area of interest
            ex_inputs_hist = []  # historical external knowledge inputs

            flow_inputs_curr = []  # flow inputs of current day
            transition_inputs_curr = []  # transition inputs of current day
            ex_inputs_curr = []  # external knowledge inputs of current day

            assert num_weeks_hist >= 0 and num_days_hist >= 1
            """ set the start time interval to sample the data"""
            s1 = num_days_hist * self.parameters.time_interval_daily + num_intervals_before_predict
            s2 = num_weeks_hist * 7 * self.parameters.time_interval_daily + num_intervals_before_predict
            time_start = max(s1, s2)
            time_end = flow_data.shape[0]

            data_shape = flow_data.shape

            for t in range(time_start, time_end):
                if t % 100 == 0:
                    print("Currently at {} interval...".format(t))

                for x in range(flow_data.shape[1]):
                    for y in range(flow_data.shape[2]):

                        """ initialize the array to hold the samples of each node at each time interval """
                        flow_inputs_hist_sample = []
                        transition_inputs_hist_sample = []
                        ex_inputs_hist_sample = []

                        flow_inputs_curr_sample = []
                        transition_inputs_curr_sample = []
                        ex_inputs_curr_sample = []

                        """ start the samplings of previous weeks """
                        for week_cnt in range(num_weeks_hist):
                            this_week_start_time = int(t - (
                                    num_weeks_hist - week_cnt) * 7 * self.parameters.time_interval_daily - num_intervals_before_predict)

                            for int_cnt in range(num_intervals_hist):
                                t_now = this_week_start_time + int_cnt

                                local_flow = flow_data[t_now, :, :, :]

                                local_trans = np.zeros((data_shape[0], data_shape[1], 4),
                                                       dtype=np.float32)

                                local_trans[:, :, 0] = \
                                    trans_data[0, t_now, :, :, x, y]
                                local_trans[:, :, 1] = \
                                    trans_data[1, t_now, :, :, x, y]
                                local_trans[:, :, 2] = \
                                    trans_data[0, t_now, x, y, :, :]
                                local_trans[:, :, 3] = \
                                    trans_data[1, t_now, x, y, :, :]

                                flow_inputs_hist_sample.append(local_flow)
                                transition_inputs_hist_sample.append(local_trans)
                                ex_inputs_hist_sample.append(ex_knlg_data[t_now, :])

                        """ start the samplings of previous days"""
                        for hist_day_cnt in range(num_days_hist):
                            """ define the start time in previous days """
                            hist_day_start_time = int(t - (
                                    num_days_hist - hist_day_cnt) * self.parameters.time_interval_daily - num_intervals_before_predict)

                            """ generate samples from the previous days """
                            for int_cnt in range(num_intervals_hist):
                                t_now = hist_day_start_time + int_cnt

                                local_flow = flow_data[t_now, :, :, :]

                                local_trans = np.zeros((data_shape[0], data_shape[1], 4),
                                                       dtype=np.float32)

                                local_trans[:, :, 0] = \
                                    trans_data[0, t_now, :, :, x, y]
                                local_trans[:, :, 1] = \
                                    trans_data[1, t_now, :, :, x, y]
                                local_trans[:, :, 2] = \
                                    trans_data[0, t_now, x, y, :, :]
                                local_trans[:, :, 3] = \
                                    trans_data[1, t_now, x, y, :, :]

                                flow_inputs_hist_sample.append(local_flow)
                                transition_inputs_hist_sample.append(local_trans)
                                ex_inputs_hist_sample.append(ex_knlg_data[t_now, :])

                        """ sampling of inputs of current day, the details are similar to those mentioned above """
                        for int_cnt in range(num_intervals_curr):
                            t_now = int(t - (num_intervals_curr - int_cnt))

                            local_flow = flow_data[t_now, :, :, :]

                            local_trans = np.zeros((data_shape[0], data_shape[1], 4),
                                                   dtype=np.float32)

                            local_trans[:, :, 0] = \
                                trans_data[0, t_now, :, :, x, y]
                            local_trans[:, :, 1] = \
                                trans_data[1, t_now, :, :, x, y]
                            local_trans[:, :, 2] = \
                                trans_data[0, t_now, x, y, :, :]
                            local_trans[:, :, 3] = \
                                trans_data[1, t_now, x, y, :, :]

                            flow_inputs_curr_sample.append(local_flow)
                            transition_inputs_curr_sample.append(local_trans)
                            ex_inputs_curr_sample.append(ex_knlg_data[t_now, :])

                        """ append the samples of each node to the overall inputs arrays """
                        flow_inputs_curr.append(np.array(flow_inputs_curr_sample, dtype=np.float32))
                        transition_inputs_curr.append(np.array(transition_inputs_curr_sample, dtype=np.float32))
                        ex_inputs_curr.append(np.array(ex_inputs_curr_sample, dtype=np.float32))
                        flow_inputs_hist.append(np.array(flow_inputs_hist_sample, dtype=np.float32))
                        transition_inputs_hist.append(np.array(transition_inputs_hist_sample, dtype=np.float32))
                        ex_inputs_hist.append(np.array(ex_inputs_hist_sample, dtype=np.float32))

                        xs.append(flow_data[t - 1, x, y, :])

                        """ generating the ground truth for each sample """
                        ys.append(flow_data[t, x, y, :])

                        tar_t = np.zeros((2 * local_block_len_half + 1, 2 * local_block_len_half + 1, 4),
                                         dtype=np.float32)

                        tar_t[:, :, 0] = \
                            trans_data[0, t, :, :, x, y]
                        tar_t[:, :, 1] = \
                            trans_data[1, t, :, :, x, y]
                        tar_t[:, :, 2] = \
                            trans_data[0, t, x, y, :, :]
                        tar_t[:, :, 3] = \
                            trans_data[1, t, x, y, :, :]

                        ys_transitions.append(tar_t)

            """ convert the inputs arrays to matrices """
            flow_inputs_curr = np.array(flow_inputs_curr, dtype=np.float32).transpose((0, 2, 3, 1, 4))
            transition_inputs_curr = np.array(transition_inputs_curr, dtype=np.float32).transpose((0, 2, 3, 1, 4))
            ex_inputs_curr = np.array(ex_inputs_curr, dtype=np.float32)
            flow_inputs_hist = np.array(flow_inputs_hist, dtype=np.float32).transpose((0, 2, 3, 1, 4))
            transition_inputs_hist = np.array(transition_inputs_hist, dtype=np.float32).transpose((0, 2, 3, 1, 4))
            ex_inputs_hist = np.array(ex_inputs_hist, dtype=np.float32)

            xs = np.array(xs, dtype=np.float32)

            ys = np.array(ys, dtype=np.float32)
            ys_transitions = np.array(ys_transitions, dtype=np.float32)

            """ save the matrices """
            np.savez_compressed("data/flow_inputs_curr_{}_{}.npz".format(self.dataset, datatype), data=flow_inputs_curr)
            np.savez_compressed("data/transition_inputs_curr_{}_{}.npz".format(self.dataset, datatype),
                                data=transition_inputs_curr)
            np.savez_compressed("data/ex_inputs_curr_{}_{}.npz".format(self.dataset, datatype), data=ex_inputs_curr)
            np.savez_compressed("data/flow_inputs_hist_{}_{}.npz".format(self.dataset, datatype), data=flow_inputs_hist)
            np.savez_compressed("data/transition_inputs_hist_{}_{}.npz".format(self.dataset, datatype),
                                data=transition_inputs_hist)
            np.savez_compressed("data/ex_inputs_hist_{}_{}.npz".format(self.dataset, datatype), data=ex_inputs_hist)
            np.savez_compressed("data/xs_{}_{}.npz".format(self.dataset, datatype), data=xs)
            np.savez_compressed("data/ys_{}_{}.npz".format(self.dataset, datatype), data=ys)
            np.savez_compressed("data/ys_transitions_{}_{}.npz".format(self.dataset, datatype), data=ys_transitions)

            return flow_inputs_hist, transition_inputs_hist, ex_inputs_hist, flow_inputs_curr, transition_inputs_curr, ex_inputs_curr, xs, ys_transitions, ys
