f_train = "./data/NYTaxi/flow_train.npz"
f_test = "./data/NYTaxi/flow_test.npz"
t_train = "./data/NYTaxi/trans_train.npz"
t_test = "./data/NYTaxi/trans_test.npz"
ex_train = "./data/NYTaxi/ex_knlg_train.npz"
ex_test = "./data/NYTaxi/ex_knlg_test.npz"
f_train_max = 1518.0
f_test_max = 1518.0
t_train_max = 186.0
t_test_max = 186.0
n_sec_int = 1800
n_int_day = 48
total_day = 60
n_int = total_day * 24 * 60 * 60 / n_sec_int
loss_threshold = 10
len_r = 16
len_c = 12

if __name__ == "__main__":
    import numpy as np
    f_data_train = np.load(f_train)['flow']
    f_data_test = np.load(f_test)['flow']
    t_data_train = np.load(t_train)['trans']
    t_data_test = np.load(t_test)['trans']
    ex_data_train = np.load(ex_train)['external_knowledge']
    ex_data_test = np.load(ex_test)['external_knowledge']
    np.savez_compressed(f_train, data=f_data_train)
    np.savez_compressed(f_test, data=f_data_test)
    np.savez_compressed(t_train, data=t_data_train)
    np.savez_compressed(t_test, data=t_data_test)
    np.savez_compressed(ex_train, data=ex_data_train)
    np.savez_compressed(ex_test, data=ex_data_test)