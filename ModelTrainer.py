from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

import time

import parameters_nyctaxi
import parameters_nycbike

from models import STSAN_XL
from utils.CustomSchedule import CustomSchedule
from utils.EarlystopHelper import EarlystopHelper
from utils.ReshuffleHelper import ReshuffleHelper
from utils.tools import DatasetGenerator, write_result, create_look_ahead_mask
from utils.Metrics import MAE

""" use mirrored strategy for distributed training """
strategy = tf.distribute.MirroredStrategy()
print('Number of GPU devices: {}'.format(strategy.num_replicas_in_sync))


class ModelTrainer:
    def __init__(self, model_index, args):
        assert args.dataset == 'taxi' or args.dataset == 'bike'
        self.model_index = model_index
        self.args = args
        self.GLOBAL_BATCH_SIZE = args.BATCH_SIZE * strategy.num_replicas_in_sync
        self.dataset_generator = DatasetGenerator(args.d_model,
                                                  args.dataset,
                                                  self.GLOBAL_BATCH_SIZE,
                                                  args.n_hist_week,
                                                  args.n_hist_day,
                                                  args.n_hist_int,
                                                  args.n_curr_int,
                                                  args.n_int_before,
                                                  args.n_pred,
                                                  args.test_model)

        if args.dataset == 'taxi':
            self.t_max = parameters_nyctaxi.t_train_max
            self.f_max = parameters_nyctaxi.f_train_max
            self.es_patiences = [5, args.es_patience]
            self.es_threshold = args.es_threshold
            self.reshuffle_threshold = [0.8, 1.3, 1.7]
            self.test_threshold = 10 / self.f_max
        else:
            self.t_max = parameters_nycbike.t_train_max
            self.f_max = parameters_nycbike.f_train_max
            self.es_patiences = [5, args.es_patience]
            self.es_threshold = args.es_threshold
            self.reshuffle_threshold = [0.8, 1.3, 1.7]
            self.test_threshold = 10 / self.f_max

    def train(self):
        args = self.args
        test_model = args.test_model
        result_output_path = "results/stsan_xl/{}.txt".format(self.model_index)

        train_dataset, val_dataset = self.dataset_generator.build_dataset('train', args.load_saved_data, strategy)
        test_dataset = self.dataset_generator.build_dataset('test', args.load_saved_data, strategy)

        with strategy.scope():

            def tf_summary_scalar(summary_writer, name, value, step):
                with summary_writer.as_default():
                    tf.summary.scalar(name, value, step=step)

            def print_verbose(epoch, final_test):
                if final_test:
                    template_rmse = "RMSE(in/out):"
                    template_mae = "MAE(in/out):"
                    for i in range(args.n_pred):
                        template_rmse += ' {}. {:.2f}({:.6f})/{:.2f}({:.6f})'.format(
                            i + 1,
                            in_rmse_test[i].result() * self.f_max,
                            in_rmse_test[i].result(),
                            out_rmse_test[i].result() * self.f_max,
                            out_rmse_test[i].result()
                        )
                        template_mae += ' {}. {:.2f}({:.6f})/{:.2f}({:.6f})'.format(
                            i + 1,
                            in_mae_test[i].result() * self.f_max,
                            in_mae_test[i].result(),
                            out_mae_test[i].result() * self.f_max,
                            out_mae_test[i].result()
                        )
                    template = "Final:\n" + template_rmse + "\n" + template_mae + "\n\n"
                    write_result(result_output_path, template)
                else:
                    template = "Epoch {} RMSE(in/out):".format(epoch + 1)
                    for i in range(args.n_pred):
                        template += " {}. {:.6f}/{:.6f}".format\
                            (i + 1, in_rmse_test[i].result(), out_rmse_test[i].result())
                    template += "\n\n"
                    write_result(result_output_path,
                                 'Validation Result (Min-Max Norm, filtering out trivial grids):\n' + template, False)
                    print(template)

            loss_object = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

            def loss_function(real, pred):
                loss_ = loss_object(real, pred)
                return tf.nn.compute_average_loss(loss_, global_batch_size=self.GLOBAL_BATCH_SIZE)

            in_rmse_train = tf.keras.metrics.RootMeanSquaredError(dtype=tf.float32)
            out_rmse_train = tf.keras.metrics.RootMeanSquaredError(dtype=tf.float32)
            in_rmse_test = [tf.keras.metrics.RootMeanSquaredError(dtype=tf.float32) for _ in range(args.n_pred)]
            out_rmse_test = [tf.keras.metrics.RootMeanSquaredError(dtype=tf.float32) for _ in range(args.n_pred)]

            in_mae_test = [MAE() for _ in range(args.n_pred)]
            out_mae_test = [MAE() for _ in range(args.n_pred)]

            learning_rate = CustomSchedule(args.d_model, args.warmup_steps)

            optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
            # if args.mixed_precision:
            #     optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)

            stsan_xl = STSAN_XL(args.num_layers,
                                args.d_model,
                                args.d_global,
                                args.num_heads,
                                args.dff,
                                args.cnn_layers,
                                args.cnn_filters,
                                args.seq_len,
                                args.dropout_rate)

            checkpoint_path = "./checkpoints/stsan_xl/{}".format(self.model_index)

            ckpt = tf.train.Checkpoint(STSAN_XL=stsan_xl, optimizer=optimizer)

            ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path,
                                                      max_to_keep=(args.es_patience + self.es_patiences[0] + 1))

            if ckpt_manager.latest_checkpoint:
                if len(ckpt_manager.checkpoints) <= args.es_patience:
                    ckpt.restore(ckpt_manager.checkpoints[-1])
                elif len(ckpt_manager.checkpoints) < args.es_patience + self.es_patiences[0] + 1:
                    ckpt.restore(ckpt_manager.checkpoints[args.es_patience - 1])
                else:
                    ckpt.restore(ckpt_manager.checkpoints[0])
                print('Latest checkpoint restored!!')

            def train_step(inp_ft, inp_ex, dec_inp_f, dec_inp_ex, cors, y):

                lh_mask = create_look_ahead_mask(tf.shape(y)[1])

                with tf.GradientTape() as tape:
                    predictions, _ = stsan_xl(inp_ft, inp_ex, dec_inp_f, dec_inp_ex, cors, True, lh_mask)
                    loss = loss_function(y, predictions)

                gradients = tape.gradient(loss, stsan_xl.trainable_variables)
                optimizer.apply_gradients(zip(gradients, stsan_xl.trainable_variables))

                in_rmse_train(y[:, :, 0], predictions[:, :, 0])
                out_rmse_train(y[:, :, 1], predictions[:, :, 1])

                return loss

            @tf.function
            def distributed_train_step(inp_ft, inp_ex, dec_inp_f, dec_inp_ex, cors, y):
                per_replica_losses = strategy.experimental_run_v2 \
                    (train_step, args=(inp_ft, inp_ex, dec_inp_f, dec_inp_ex, cors, y,))

                return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

            def test_step(inp_ft, inp_ex, dec_inp_f, dec_inp_ex, cors, y, final_test=False):
                tar_inp = dec_inp_f[:, :1, :]
                for i in range(args.n_pred):
                    tar_inp_ex = dec_inp_ex[:, :i + 1, :]
                    predictions, _ = stsan_xl(inp_ft, inp_ex, tar_inp, tar_inp_ex, cors, training=False)

                    """ here we filter out all nodes where their real flows are less than 10 """
                    real_in = y[:, i, 0]
                    real_out = y[:, i, 1]
                    pred_in = predictions[:, -1, 0]
                    pred_out = predictions[:, -1, 1]
                    mask_in = tf.where(tf.math.greater(real_in, self.test_threshold))
                    mask_out = tf.where(tf.math.greater(real_out, self.test_threshold))
                    masked_real_in = tf.gather_nd(real_in, mask_in)
                    masked_real_out = tf.gather_nd(real_out, mask_out)
                    masked_pred_in = tf.gather_nd(pred_in, mask_in)
                    masked_pred_out = tf.gather_nd(pred_out, mask_out)
                    in_rmse_test[i](masked_real_in, masked_pred_in)
                    out_rmse_test[i](masked_real_out, masked_pred_out)
                    if final_test:
                        in_mae_test[i](masked_real_in, masked_pred_in)
                        out_mae_test[i](masked_real_out, masked_pred_out)

                    tar_inp = tf.concat([tar_inp, predictions[:, -1:, :]], axis=-2)

            @tf.function
            def distributed_test_step(inp_ft, inp_ex, dec_inp_f, dec_inp_ex, cors, y, final_test=False):
                return strategy.experimental_run_v2(test_step, args=(
                    inp_ft, inp_ex, dec_inp_f, dec_inp_ex, cors, y, final_test,))

            def evaluate(eval_dataset, epoch, verbose=1, final_test=False):
                for i in range(args.n_pred):
                    in_rmse_test[i].reset_states()
                    out_rmse_test[i].reset_states()

                for (batch, (inp, tar)) in enumerate(eval_dataset):
                    inp_ft = inp["inp_ft"]
                    inp_ex = inp["inp_ex"]
                    dec_inp_f = inp["dec_inp_f"]
                    dec_inp_ex = inp["dec_inp_ex"]
                    cors = inp["cors"]

                    y = tar["y"]

                    distributed_test_step(inp_ft, inp_ex, dec_inp_f, dec_inp_ex, cors, y, final_test)

                if verbose:
                    print_verbose(epoch, final_test)

            """ Start training... """
            write_result(result_output_path, "\nStart training...\n")
            es_flag = False
            check_flag = False
            es_helper = EarlystopHelper(self.es_patiences, self.es_threshold)
            reshuffle_helper = ReshuffleHelper(self.es_patiences[1], self.reshuffle_threshold)
            summary_writer = tf.summary.create_file_writer('./tensorboard/stsan_xl/{}'.format(self.model_index))
            step_cnt = 0
            for epoch in range(args.MAX_EPOCH):

                start = time.time()

                in_rmse_train.reset_states()
                out_rmse_train.reset_states()

                for (batch, (inp, tar)) in enumerate(train_dataset):

                    inp_ft = inp["inp_ft"]
                    inp_ex = inp["inp_ex"]
                    dec_inp_f = inp["dec_inp_f"]
                    dec_inp_ex = inp["dec_inp_ex"]
                    cors = inp["cors"]

                    y = tar["y"]

                    total_loss = distributed_train_step(inp_ft, inp_ex, dec_inp_f, dec_inp_ex, cors, y)

                    step_cnt += 1
                    tf_summary_scalar(summary_writer, "total_loss", total_loss, step_cnt)

                    if (batch + 1) % 100 == 0 and args.verbose_train:
                        print('Epoch {} Batch {} in_rmse {:.6f} out_rmse {:.6f}'.format(
                            epoch + 1, batch + 1, in_rmse_train.result(), out_rmse_train.result()))

                if args.verbose_train:
                    template = 'Epoch {} in_RMSE {:.6f} out_RMSE {:.6f}\n'.format\
                        (epoch + 1, in_rmse_train.result(), out_rmse_train.result())
                    write_result(result_output_path, template)
                    tf_summary_scalar(summary_writer, "in_rmse_train", in_rmse_train.result(), epoch + 1)
                    tf_summary_scalar(summary_writer, "out_rmse_train", out_rmse_train.result(), epoch + 1)

                eval_rmse = (in_rmse_train.result() + out_rmse_train.result()) / 2

                if check_flag == False and es_helper.refresh_status(eval_rmse):
                    check_flag = True

                if test_model or check_flag:
                    print("Validation Result (Min-Max Norm, filtering out trivial grids): ")
                    evaluate(val_dataset, epoch, final_test=False)
                    tf_summary_scalar(summary_writer, "in_rmse_test", in_rmse_test[0].result(), epoch + 1)
                    tf_summary_scalar(summary_writer, "out_rmse_test", out_rmse_test[0].result(), epoch + 1)
                    es_flag = es_helper.check(in_rmse_test[0].result() + out_rmse_test[0].result(), epoch)
                    tf_summary_scalar(summary_writer, "best_epoch", es_helper.get_bestepoch(), epoch + 1)

                ckpt_save_path = ckpt_manager.save()
                print('Saving checkpoint for epoch {} at {}\n'.format(epoch + 1, ckpt_save_path))

                if es_flag:
                    print("Early stoping...")
                    ckpt.restore(ckpt_manager.checkpoints[- args.es_patience - 1])
                    print('Checkpoint restored!! At epoch {}\n'.format(int(epoch - args.es_patience)))
                    break

                if test_model or reshuffle_helper.check(epoch):
                    train_dataset, val_dataset = \
                        self.dataset_generator.build_dataset('train', args.load_saved_data, strategy)

                tf_summary_scalar(summary_writer, "epoch_time", time.time() - start, epoch + 1)
                print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

                if test_model:
                    break

            write_result(result_output_path, "Start testing (filtering out trivial grids):\n")
            evaluate(test_dataset, epoch, final_test=True)
            tf_summary_scalar(summary_writer, "final_in_rmse", in_rmse_test[0].result(), 1)
            tf_summary_scalar(summary_writer, "final_out_rmse", out_rmse_test[0].result(), 1)
