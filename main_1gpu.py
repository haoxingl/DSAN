from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os
import shutil
import numpy as np
from utils.tools import ResultWriter

parser = argparse.ArgumentParser(description='Hyperparameters')
parser.add_argument('--dataset', default='ctm', help='taxi or bike or ctm')
parser.add_argument('--gpu_ids', default='4', help='indexes of gpus to use')
parser.add_argument('--memory_growth', default=False, help='allow memory growth')
parser.add_argument('--index', default='64x1', help='model index')
parser.add_argument('--test_name', default=None, help='for fine tuning')
parser.add_argument('--hyp', default=[None], help='for fine tuning')
parser.add_argument('--run_time', default=10, help='indexes of gpus to use')
parser.add_argument('--remove_old_files', default=True, help='remove old results, checkpoints, and tensorboard')
parser.add_argument('--load_saved_data', default=False, help='load saved data sets')
parser.add_argument('--no_save', default=False, help='for dev only')
parser.add_argument('--test_model', default=None, help='for dev only')
parser.add_argument('--mixed_precision', default=False, help='enable mixed precision')
parser.add_argument('--always_test', default=None, help='for dev only')

""" Model hyperparameters """
d_model = 64
parser.add_argument('--n_layer', default=3, help='num of self-attention layers')
parser.add_argument('--d_model', default=d_model, help='model dimension')
parser.add_argument('--dff', default=d_model * 4, help='dimension of feed-forward networks')
parser.add_argument('--n_head', default=8, help='number of attention heads')
parser.add_argument('--r_d', default=0.1, help='dropout rate')
parser.add_argument('--conv_layer', default=3, help='number of projection layer')
parser.add_argument('--conv_filter', default=d_model, help='dimension of projection later')

""" Training settings """
weights_t = np.array([1 for _ in range(12)], dtype=np.float32)[:, np.newaxis]   # Joint training weights for time steps
weights_f = np.array([1 for _ in range(2)], dtype=np.float32)[np.newaxis, :]    # Joint training weights for features
weights = None
# weights = weights_t * weights_f
# weights[0, :] = weights[0, :] * 12 * 0.5
# weights[1:, :] = weights[1:, :] * 12 * (1 - 0.5)/11
parser.add_argument('--MAX_EPOCH', default=200, help='Max epoch')
parser.add_argument('--BATCH_SIZE', default=64, help='batch size for each GPU')
parser.add_argument('--warmup_steps', default=64000, help='warm up step')
parser.add_argument('--verbose_train', default=1, help='1: enable verbose, 0: disable verbose')
parser.add_argument('--weights', default=weights, help='joint training weights')
parser.add_argument('--es_patience', default=5, help='early stop patience (epoch)')
parser.add_argument('--es_threshold', default=0.01, help='for early stop helper')
parser.add_argument('--es_epoch', default=60, help='epoch after which to start early stop')
parser.add_argument('--model_summary', default=True, help='print model summary')

""" Data hyperparameters """
parser.add_argument('--n_w', default=1, help='num of previous weeks to consider')
parser.add_argument('--n_d', default=3, help='num of previous days to consider')
parser.add_argument('--n_wd_times', default=1, help='num of time in previous days to consider')
parser.add_argument('--n_p', default=1, help='num of time in today to consider')
parser.add_argument('--n_before', default=0, help='num of time before predicted time to consider')
parser.add_argument('--n_pred', default=12, help='future time to predict')
parser.add_argument('--l_half', default=3, help='determine size of DAE local block')
parser.add_argument('--l_half_g', default=5, help='used for limiting size of global input')
parser.add_argument('--pre_shuffle', default=True, help='shuffle data before TF data sets')
parser.add_argument('--same_padding', default=False, help='use same_padding instead of zero_padding')
parser.add_argument('--st_revert', default=False, help='revert spatial and temporal axes of input data')

args = parser.parse_args()

def remove_oldfiles(model_index):
    try:
        shutil.rmtree('checkpoints/{}'.format(model_index), ignore_errors=True)
    except:
        pass
    try:
        os.remove('results/{}.txt'.format(model_index))
    except:
        pass
    try:
        shutil.rmtree(os.environ['HOME'] + '/tensorboard/dsan/{}'.format(model_index), ignore_errors=True)
    except:
        pass

if args.mixed_precision:
    os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
if args.gpu_ids:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

assert args.dataset in ['taxi', 'bike', 'ctm']
print("Dataset chosen: {}".format(args.dataset))

from train import TrainModel
import tensorflow as tf
import tensorflow.keras.backend as K

gpus = tf.config.list_physical_devices('GPU')

if gpus and args.memory_growth:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

if not os.path.exists('results'):
    os.makedirs('results')

if __name__ == "__main__":
    if args.test_name:
        for this_arg in args.hyp:
            for cnt in range(1 if args.test_model else args.run_time):
                model_index = args.dataset + '_{}_{}_{}_{}'.format(
                    'test' if args.test_model else args.index, args.test_name, this_arg, cnt + 1)
                print('Model index: {}'.format(model_index))
                result_writer = ResultWriter("results/{}.txt".format(model_index))

                exec("%s = %d" % ('args.{}'.format(args.test_name), this_arg))

                if args.remove_old_files:
                    remove_oldfiles(model_index)

                result_writer.write(str(args))

                model_trainer = TrainModel(model_index, args)
                print("\nStrat training DSAN...\n")
                model_trainer.train()

                args.load_saved_data = True
                K.clear_session()

                if args.test_model:
                    remove_oldfiles(model_index)

    else:
        for cnt in range(1 if args.test_model else args.run_time):
            model_index = args.dataset + '_{}_{}'.format('test' if args.test_model else args.index, cnt + 1)
            print('Model index: {}'.format(model_index))
            result_writer = ResultWriter("results/{}.txt".format(model_index))

            if args.remove_old_files:
                remove_oldfiles(model_index)

            result_writer.write(str(args))

            model_trainer = TrainModel(model_index, args)
            print("\nStrat training DSAN...\n")
            model_trainer.train()

            args.load_saved_data = True
            K.clear_session()

            if args.test_model:
                remove_oldfiles(model_index)
