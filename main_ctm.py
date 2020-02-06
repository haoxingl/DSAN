from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os
import shutil
from utils.tools import write_result

parser = argparse.ArgumentParser(description='Hyperparameters')
parser.add_argument('--dataset', default='ctm', help='taxi or bike or ctm')
parser.add_argument('--gpu_ids', default='7', help='indexes of gpus to use')
parser.add_argument('--index', default=9, help='indexes of model to be trained')
parser.add_argument('--test_name', default="ctm")
parser.add_argument('--hyp', default=[1])
parser.add_argument('--run_time', default=2)
parser.add_argument('--BATCH_SIZE', default=128)
parser.add_argument('--local_block_len', default=3)
parser.add_argument('--local_block_len_g', default=5)
parser.add_argument('--remove_old_files', default=True)
parser.add_argument('--load_saved_data', default=False)
parser.add_argument('--no_save', default=False)
parser.add_argument('--es_patience', default=10)
parser.add_argument('--es_threshold', default=0.01)
parser.add_argument('--test_model', default=100)
parser.add_argument('--mixed_precision', default=False)
parser.add_argument('--always_test', default=None)
parser.add_argument('--trace_graph', default=False)
parser.add_argument('--gm_growth', default=False)

""" Model hyperparameters """
d_model = 64
weight_1 = None
parser.add_argument('--num_layers', default=3, help='num of self-attention layers')
parser.add_argument('--d_model', default=d_model, help='model dimension')
parser.add_argument('--dff', default=d_model * 4, help='dimension of feed-forward networks')
parser.add_argument('--num_heads', default=8, help='number of attention heads')
parser.add_argument('--dropout_rate', default=0.1)
parser.add_argument('--cnn_layers', default=3)
parser.add_argument('--cnn_filters', default=d_model)
parser.add_argument('--weight_1', default=weight_1)

""" Training settings """
parser.add_argument('--MAX_EPOCH', default=250)
parser.add_argument('--warmup_steps', default=4000)
parser.add_argument('--verbose_train', default=1)

""" Data hyperparameters """
parser.add_argument('--n_hist_week', default=1, help='num of previous weeks to consider')
parser.add_argument('--n_hist_day', default=3, help='num of previous days to consider')
parser.add_argument('--n_hist_int', default=1, help='num of time in previous days to consider')
parser.add_argument('--n_curr_int', default=1, help='num of time in today to consider')
parser.add_argument('--n_int_before', default=0, help='num of time before predicted time to consider')
parser.add_argument('--n_pred', default=12, help='future time to predict')
parser.add_argument('--st_revert', default=False)

args = parser.parse_args()


def write_args(args, m_ind):
    result_output_path = "results/stsan_xl/{}.txt".format(m_ind)

    write_result(result_output_path, str(args))

def remove_oldfiles(model_index):
    try:
        shutil.rmtree('./checkpoints/stsan_xl/{}'.format(model_index), ignore_errors=True)
    except:
        pass
    try:
        os.remove('./results/stsan_xl/{}.txt'.format(model_index))
    except:
        pass
    try:
        shutil.rmtree('/home/lxx/tensorboard/stsan_xl/{}'.format(model_index), ignore_errors=True)
    except:
        pass

if args.mixed_precision:
    os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

assert args.dataset in ['taxi', 'bike', 'ctm']
print("Dataset chosen: {}".format(args.dataset))

from ModelTrainer import ModelTrainer
import tensorflow.keras.backend as K

if not os.path.exists('./results/stsan_xl'):
    os.makedirs('./results/stsan_xl')

if __name__ == "__main__":
    if args.test_name:
        for this_arg in args.hyp:
            for cnt in range(args.run_time):
                model_index = args.dataset + '_{}_{}_{}_{}'.format(args.index, args.test_name, this_arg, cnt + 1)
                print('Model index: {}'.format(model_index))

                exec("%s = %d" % ('args.{}'.format(args.test_name), this_arg))

                if args.remove_old_files:
                    remove_oldfiles(model_index)

                write_args(args, model_index)

                model_trainer = ModelTrainer(model_index, args)
                print("\nStrat training STSAN-XL...\n")
                model_trainer.train()

                args.load_saved_data = True
                K.clear_session()

            args.load_saved_data = False
    else:
        for cnt in range(args.run_time):
            model_index = args.dataset + '_{}_{}'.format(args.index, cnt + 1)
            print('Model index: {}'.format(model_index))

            if args.remove_old_files:
                remove_oldfiles(model_index)

            write_args(args, model_index)

            model_trainer = ModelTrainer(model_index, args)
            print("\nStrat training STSAN-XL...\n")
            model_trainer.train()

            args.load_saved_data = True
            K.clear_session()
