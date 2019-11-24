from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os
import shutil

parser = argparse.ArgumentParser(description='Hyperparameters')
parser.add_argument('--dataset', default='taxi', help='taxi or bike')
parser.add_argument('--gpu_ids', default='0, 1, 2, 3, 4, 5, 6, 7', help='indexes of gpus to use')
parser.add_argument('--model_indexes', default=[1, 2], help='indexes of model to be trained')

""" Model hyperparameters """
parser.add_argument('--num_layers', default=4, help='num of self-attention layers')
parser.add_argument('--d_model', default=64, help='model dimension')
parser.add_argument('--d_global', default=64, help='model dimension')
parser.add_argument('--dff', default=128, help='dimension of feed-forward networks')
parser.add_argument('--d_final', default=256, help='dimension of final output dense layer')
parser.add_argument('--num_heads', default=8, help='number of attention heads')
parser.add_argument('--dropout_rate', default=0.1)
parser.add_argument('--cnn_layers', default=3)
parser.add_argument('--cnn_filters', default=64)
parser.add_argument('--weight_f_in', default=0.4)
parser.add_argument('--weight_f_out', default=0.6)
parser.add_argument('--weight_f', default=0.5)
parser.add_argument('--weight_t', default=0.5)
parser.add_argument('--weight_in', default=0.4)
parser.add_argument('--weight_out', default=0.6)

""" Training settings """
parser.add_argument('--remove_old_files', default=True)
parser.add_argument('--MAX_EPOCH', default=500)
parser.add_argument('--BATCH_SIZE', default=16)
parser.add_argument('--es_patience', default=10)
parser.add_argument('--es_threshold', default=0.01)
parser.add_argument('--warmup_steps', default=4000)
parser.add_argument('--verbose_train', default=1)

""" Data hyperparameters """
n_hist_week = 1
n_hist_day = 3
n_hist_int = 1
n_curr_int = 3
n_int_before = 0
seq_len = (n_hist_week + n_hist_day) * n_hist_int + n_curr_int
parser.add_argument('--load_saved_data', default=False)
parser.add_argument('--n_hist_week', default=n_hist_week, help='num of previous weeks to consider')
parser.add_argument('--n_hist_day', default=n_hist_day, help='num of previous days to consider')
parser.add_argument('--n_hist_int', default=n_hist_int, help='num of time in previous days to consider')
parser.add_argument('--n_curr_int', default=n_curr_int, help='num of time in today to consider')
parser.add_argument('--n_int_before', default=1, help='num of time before predicted time to consider')
parser.add_argument('--seq_len', default=seq_len, help='total length of historical data')
parser.add_argument('--n_pred', default=5, help='future time to predict')

args = parser.parse_args()

print("num_layers: {}, d_model: {}, dff: {}, num_heads: {}, cnn_layers: {}, cnn_filters: {}" \
      .format(args.num_layers,
              args.d_model,
              args.dff,
              args.num_heads,
              args.cnn_layers,
              args.cnn_filters
              ))
print(
    "BATCH_SIZE: {}, es_patience: {}".format(
        args.BATCH_SIZE, args.es_patience))
print(
    "n_hist_week: {}, n_hist_day: {}, n_hist_int: {}, n_curr_int: {}, n_int_before: {}, n_pred: {}" \
        .format(args.n_hist_week,
                args.n_hist_day,
                args.n_hist_int,
                args.n_curr_int,
                args.n_int_before,
                args.n_pred
                ))

# os.environ['F_ENABLE_AUTO_MIXED_PRECISION'] = '1'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

assert args.dataset == 'taxi' or args.dataset == 'bike'
print("Dataset chosen: {}".format(args.dataset))

from ModelTrainer import ModelTrainer

if __name__ == "__main__":
    for index in range(args.model_indexes[0], args.model_indexes[1]):
        model_index = args.dataset + '_{}'.format(index)
        print('Model index: {}'.format(model_index))
        if args.remove_old_files:
            try:
                shutil.rmtree('./checkpoints/stsan_xl/{}'.format(model_index), ignore_errors=True)
            except:
                pass
            try:
                if not os.path.exists('./results/stsan_xl'):
                    os.makedirs('./results/stsan_xl')
                os.remove('./results/stsan_xl/{}.txt'.format(model_index))
            except:
                pass
            try:
                shutil.rmtree('./tensorboard/stsan_xl/{}'.format(model_index), ignore_errors=True)
            except:
                pass
        model_trainer = ModelTrainer(model_index, args)
        print("\nStrat training STSAN-XL...\n")
        model_trainer.train()
