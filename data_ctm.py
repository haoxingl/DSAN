import numpy as np
from utils.CordinateGenerator import CordinateGenerator

from data_parameters import data_parameters

data_train_all = np.load('./data/ctm_old/ctm_train.npz')
data_test_all = np.load('./data/ctm_old/ctm_test.npz')
data_train = data_train_all['data']
data_test = data_test_all['data']
data = np.concatenate((data_train, data_test), axis=0)
data_du = data_train[..., 0]/60
data_re = data_train[..., 1]
print(data_du.max())
print(data_du[(data_du >= 60)].mean())
print(data_du[(data_du >= 60)].std())
print(data_re.max())
print(data_re[(data_re >= 10)].mean())
print(data_re[(data_re >= 10)].std())
data_train[..., 0] = data_train[..., 0]/60
data_test[..., 0] = data_test[..., 0]/60
np.savez_compressed("./data/ctm/ctm_train.npz", data=data_train, ex_knlg=data_train_all['ex_knlg'])
np.savez_compressed("./data/ctm/ctm_test.npz", data=data_test, ex_knlg=data_test_all['ex_knlg'])

data_du = data[..., 0]/60
data_re = data[..., 1]
print(data_du.max())
print(data_du[(data_du >= 60)].mean())
print(data_du[(data_du >= 60)].std())
print(data_re.max())
print(data_re[(data_re >= 10)].mean())
print(data_re[(data_re >= 10)].std())


data_train = np.load('./data/ctm/ctm_train.npz')['data']
data_test = np.load('./data/ctm/ctm_test.npz')['data']
# data = np.concatenate((data_train, data_test), axis=0)
data_du = data_train[..., 0]
data_re = data_train[..., 1]
print(data_du.max())
print(data_du[(data_du >= 60)].mean())
print(data_du[(data_du >= 60)].std())
print(data_re.max())
print(data_re[(data_re >= 10)].mean())
print(data_re[(data_re >= 10)].std())
