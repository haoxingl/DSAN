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

from utils.CustomSchedule import CustomSchedule
from utils.EarlystopHelper import EarlystopHelper
from utils.ReshuffleHelper import ReshuffleHelper
# from models import ST_SAN
from models_gelu import ST_SAN
from utils.utils import DatasetGenerator, write_result
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
        self.dataset_generator = DatasetGenerator(args.dataset,
                                                  self.GLOBAL_BATCH_SIZE,
                                                  args.num_weeks_hist,
                                                  args.num_days_hist,
                                                  args.num_intervals_hist,
                                                  args.num_intervals_curr,
                                                  args.num_intervals_before_predict,
                                                  args.local_block_len)

        if args.dataset == 'taxi':
            self.trans_max = parameters_nyctaxi.trans_train_max
            self.flow_max = parameters_nyctaxi.flow_train_max
            self.earlystop_patiences_1 = [5, args.earlystop_patience_stream_t]
            self.earlystop_patiences_2 = [5, args.earlystop_patience_stsan]
            self.earlystop_thres_1 = 0.02
            self.earlystop_thres_2 = 0.01
            self.reshuffle_thres_stream_t = [0.8, 1.3]
            self.reshuffle_thres_stsan = [0.8, 1.3, 1.7]
            self.threshold = 10/self.flow_max
        else:
            self.trans_max = parameters_nycbike.trans_train_max
            self.flow_max = parameters_nycbike.flow_train_max
            self.earlystop_patiences_1 = [5, args.earlystop_patience_stream_t]
            self.earlystop_patiences_2 = [5, args.earlystop_patience_stsan]
            self.earlystop_thres_1 = 0.02
            self.earlystop_thres_2 = 0.01
            self.reshuffle_thres_stream_t = [0.8, 1.3]
            self.reshuffle_thres_stsan = [0.8, 1.3, 1.7]
            self.threshold = 10 / self.flow_max

    def train_st_san(self):
        result_output_path = "results/st_san/{}.txt".format(self.model_index)

        train_dataset, val_dataset = self.dataset_generator.load_dataset('train', self.args.load_saved_data, strategy)
        test_dataset = self.dataset_generator.load_dataset('test', self.args.load_saved_data, strategy)

        with strategy.scope():

            cnn_len = self.args.local_block_len * 2 + 1

            loss_object = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

            def loss_function(real, pred):
                loss_ = loss_object(real, pred)
                return tf.nn.compute_average_loss(loss_, global_batch_size=self.GLOBAL_BATCH_SIZE)

            train_inflow_rmse = tf.keras.metrics.RootMeanSquaredError(dtype=tf.float32)
            train_outflow_rmse = tf.keras.metrics.RootMeanSquaredError(dtype=tf.float32)
            test_inflow_rmse = tf.keras.metrics.RootMeanSquaredError(dtype=tf.float32)
            test_outflow_rmse = tf.keras.metrics.RootMeanSquaredError(dtype=tf.float32)

            # train_inflow_mae = tf.keras.metrics.MeanAbsoluteError(dtype=tf.float32)
            # train_outflow_mae = tf.keras.metrics.MeanAbsoluteError(dtype=tf.float32)
            test_inflow_mae = MAE()
            test_outflow_mae = MAE()

            learning_rate = CustomSchedule(self.args.d_model, self.args.warmup_steps)

            optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

            st_san = ST_SAN(self.args.num_layers,
                                 self.args.d_model,
                                 self.args.num_heads,
                                 self.args.dff,
                                 self.args.cnn_layers,
                                 self.args.cnn_filters,
                                 self.args.num_intervals_enc,
                                 self.args.d_final,
                                 4,
                                 self.args.dropout_rate)

            checkpoint_path = "./checkpoints/st_san/{}".format(self.model_index)

            ckpt = tf.train.Checkpoint(ST_SAN=st_san, optimizer=optimizer)

            ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path,
                                                      max_to_keep=(self.args.earlystop_patience_stsan +
                                                                   self.earlystop_patiences_2[0] + 1))

            if ckpt_manager.latest_checkpoint:
                if len(ckpt_manager.checkpoints) <= self.args.earlystop_patience_stsan:
                    ckpt.restore(ckpt_manager.checkpoints[-1])
                elif len(ckpt_manager.checkpoints) < self.args.earlystop_patience_stsan + self.earlystop_patiences_2[
                    0] + 1:
                    ckpt.restore(ckpt_manager.checkpoints[self.args.earlystop_patience_stsan - 1])
                else:
                    ckpt.restore(ckpt_manager.checkpoints[0])
                print('Latest checkpoint restored!!')

            def train_step(flow_hist, trans_hist, ex_hist, flow_curr, trans_curr, ex_curr, ys, ys_t):

                with tf.GradientTape() as tape:
                    predictions_f, predictions_t, _ = st_san(flow_hist,
                                                             trans_hist,
                                                             ex_hist,
                                                             flow_curr,
                                                             trans_curr,
                                                             ex_curr,
                                                             training=True)
                    loss_f = loss_function(ys, predictions_f)
                    loss_t = loss_function(ys_t, predictions_t)
                    final_loss = self.args.weight_f * loss_f + self.args.weight_t * loss_t

                gradients = tape.gradient(final_loss, st_san.trainable_variables)
                optimizer.apply_gradients(zip(gradients, st_san.trainable_variables))

                train_inflow_rmse(ys[:, 0], predictions_f[:, 0])
                train_outflow_rmse(ys[:, 1], predictions_f[:, 1])
                # train_inflow_mae(ys[:, 0], predictions_f[:, 0])
                # train_outflow_mae(ys[:, 1], predictions_f[:, 1])

                return loss_f, loss_t

            # train_step_signature = [
            #     tf.TensorSpec(shape=(None, None, None, None, None), dtype=tf.float32),
            #     tf.TensorSpec(shape=(None, None, None, None, None), dtype=tf.float32),
            #     tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
            #     tf.TensorSpec(shape=(None, None, None, None, None), dtype=tf.float32),
            #     tf.TensorSpec(shape=(None, None, None, None, None), dtype=tf.float32),
            #     tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
            #     tf.TensorSpec(shape=(None, None), dtype=tf.float32),
            #     tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float32)
            # ]
            #
            # @tf.function(input_signature=train_step_signature)
            @tf.function
            def distributed_train_step(flow_hist, trans_hist, ex_hist, flow_curr, trans_curr, ex_curr, ys, ys_t):
                per_replica_losses_f, per_replica_losses_t = strategy.experimental_run_v2(train_step,
                                                                                          args=(flow_hist, trans_hist, ex_hist, flow_curr, trans_curr, ex_curr, ys, ys_t,))

                return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses_f, axis=None), strategy.reduce(
                    tf.distribute.ReduceOp.SUM, per_replica_losses_t, axis=None)

            def test_step(flow_hist, trans_hist, ex_hist, flow_curr, trans_curr, ex_curr, ys, testing=False):

                predictions, _, _ = st_san(flow_hist, trans_hist, ex_hist, flow_curr, trans_curr, ex_curr,
                                           training=False)

                """ here we filter out all nodes where their real flows are less than 10 """
                real_in = ys[:, 0]
                real_out = ys[:, 1]
                pred_in = predictions[:, 0]
                pred_out = predictions[:, 1]
                mask_in = tf.where(tf.math.greater(real_in, self.threshold))
                mask_out = tf.where(tf.math.greater(real_out, self.threshold))
                masked_real_in = tf.gather_nd(real_in, mask_in)
                masked_real_out = tf.gather_nd(real_out, mask_out)
                masked_pred_in = tf.gather_nd(pred_in, mask_in)
                masked_pred_out = tf.gather_nd(pred_out, mask_out)
                test_inflow_rmse(masked_real_in, masked_pred_in)
                test_outflow_rmse(masked_real_out, masked_pred_out)
                if testing:
                    test_inflow_mae(masked_real_in, masked_pred_in)
                    test_outflow_mae(masked_real_out, masked_pred_out)

            # test_step_signature = [
            #     tf.TensorSpec(shape=(None, None, None, None, None), dtype=tf.float32),
            #     tf.TensorSpec(shape=(None, None, None, None, None), dtype=tf.float32),
            #     tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
            #     tf.TensorSpec(shape=(None, None, None, None, None), dtype=tf.float32),
            #     tf.TensorSpec(shape=(None, None, None, None, None), dtype=tf.float32),
            #     tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
            #     tf.TensorSpec(shape=(None, None), dtype=tf.float32)
            # ]
            #
            # @tf.function(input_signature=test_step_signature)
            @tf.function
            def distributed_test_step(flow_hist, trans_hist, ex_hist, flow_curr, trans_curr, ex_curr, ys, testing=False):
                return strategy.experimental_run_v2(test_step, args=(flow_hist, trans_hist, ex_hist, flow_curr, trans_curr, ex_curr, ys, testing,))

            def evaluate(eval_dataset, epoch, verbose=1, testing=False):
                test_inflow_rmse.reset_states()
                test_outflow_rmse.reset_states()
                # test_inflow_mae.reset_states()
                # test_outflow_mae.reset_states()

                for (batch, (inp, tar)) in enumerate(eval_dataset):

                    flow_hist = inp["flow_hist"]
                    trans_hist = inp["trans_hist"]
                    ex_hist = inp["ex_hist"]
                    flow_curr = inp["flow_curr"]
                    trans_curr = inp["trans_curr"]
                    ex_curr = inp["ex_curr"]

                    ys = tar["ys"]

                    distributed_test_step(flow_hist, trans_hist, ex_hist, flow_curr, trans_curr, ex_curr, ys, testing)

                    if verbose and (batch + 1) % 100 == 0:
                        if not testing:
                            print(
                                # "Epoch {} Batch {} INFLOW_RMSE {:.6f} OUTFLOW_RMSE {:.6f} INFLOW_MAE {:.6f} OUTFLOW_MAE {:.6f}".format(
                                "Epoch {} Batch {} INFLOW_RMSE {:.6f} OUTFLOW_RMSE {:.6f}".format(
                                    epoch + 1, batch + 1,
                                    test_inflow_rmse.result(),
                                    test_outflow_rmse.result()
                                    # test_inflow_mae.result(),
                                    # test_outflow_mae.result()
                                ))
                        else:
                            print(
                                "Testing: Batch {} INFLOW_RMSE {:.2f}({:.6f}) OUTFLOW_RMSE {:.2f}({:.6f}) INFLOW_MAE {:.2f}({:.6f}) OUTFLOW_MAE {:.2f}({:.6f})".format(
                                    batch + 1,
                                    test_inflow_rmse.result() * self.flow_max,
                                    test_inflow_rmse.result(),
                                    test_outflow_rmse.result() * self.flow_max,
                                    test_outflow_rmse.result(),
                                    test_inflow_mae.result() * self.flow_max,
                                    test_inflow_mae.result(),
                                    test_outflow_mae.result() * self.flow_max,
                                    test_outflow_mae.result()
                                ))

                if verbose:
                    if not testing:
                        # template = 'Epoch {} INFLOW_RMSE {:.6f} OUTFLOW_RMSE {:.6f} INFLOW_MAE {:.6f} OUTFLOW_MAE {:.6f}\n\n'.format(
                        template = 'Epoch {} INFLOW_RMSE {:.6f} OUTFLOW_RMSE {:.6f}\n\n'.format(
                            epoch + 1,
                            test_inflow_rmse.result(),
                            test_outflow_rmse.result())
                            # test_inflow_mae.result(),
                            # test_outflow_mae.result())
                        write_result(result_output_path,
                                     'Validation Result (after Min-Max Normalization, filtering out grids with flow less than consideration threshold):\n' + template)
                        print(template)
                    else:
                        template = 'Final results: INFLOW_RMSE {:.2f}({:.6f}) OUTFLOW_RMSE {:.2f}({:.6f}) INFLOW_MAE {:.2f}({:.6f}) OUTFLOW_MAE {:.2f}({:.6f})\n'.format(
                            test_inflow_rmse.result() * self.flow_max,
                            test_inflow_rmse.result(),
                            test_outflow_rmse.result() * self.flow_max,
                            test_outflow_rmse.result(),
                            test_inflow_mae.result() * self.flow_max,
                            test_inflow_mae.result(),
                            test_outflow_mae.result() * self.flow_max,
                            test_outflow_mae.result())
                        write_result(result_output_path, template)
                        print(template)

                return test_inflow_rmse.result(), test_outflow_rmse.result()

            """ Start training... """
            print('\nStart training...\n')
            write_result(result_output_path, "Start training:\n")
            earlystop_flag = False
            check_flag = False
            earlystop_helper = EarlystopHelper(self.earlystop_patiences_2, self.earlystop_thres_2)
            reshuffle_helper = ReshuffleHelper(self.earlystop_patiences_2[1], self.reshuffle_thres_stsan)
            summary_writer = tf.summary.create_file_writer('./tensorboard/st_san/{}'.format(self.model_index))
            step_cnt = 0
            for epoch in range(self.args.MAX_EPOCH):

                start = time.time()

                train_inflow_rmse.reset_states()
                train_outflow_rmse.reset_states()
                # train_inflow_mae.reset_states()
                # train_outflow_mae.reset_states()

                for (batch, (inp, tar)) in enumerate(train_dataset):

                    flow_hist = inp["flow_hist"]
                    trans_hist = inp["trans_hist"]
                    ex_hist = inp["ex_hist"]
                    flow_curr = inp["flow_curr"]
                    trans_curr = inp["trans_curr"]
                    ex_curr = inp["ex_curr"]

                    ys = tar["ys"]
                    ys_t = tar["ys_transitions"]

                    total_loss_f, total_loss_t = distributed_train_step(flow_hist, trans_hist, ex_hist, flow_curr, trans_curr, ex_curr, ys, ys_t)

                    step_cnt += 1
                    with summary_writer.as_default():
                        tf.summary.scalar("total_loss_f", total_loss_f, step=step_cnt)
                        tf.summary.scalar("total_loss_t", total_loss_t, step=step_cnt)

                    if (batch + 1) % 100 == 0 and self.args.verbose_train:
                        # print('Epoch {} Batch {} in_RMSE {:.6f} out_RMSE {:.6f} in_MAE {:.6f} out_MAE {:.6f}'.format(
                        print('Epoch {} Batch {} in_RMSE {:.6f} out_RMSE'.format(
                            epoch + 1,
                            batch + 1,
                            train_inflow_rmse.result(),
                            train_outflow_rmse.result()))
                            # train_inflow_mae.result(),
                            # train_outflow_mae.result()))

                if self.args.verbose_train:
                    # template = 'Epoch {} in_RMSE {:.6f} out_RMSE {:.6f} in_MAE {:.6f} out_MAE {:.6f}'.format(
                    template = 'Epoch {} in_RMSE {:.6f} out_RMSE {:.6f}'.format(
                        epoch + 1,
                        train_inflow_rmse.result(),
                        train_outflow_rmse.result())
                        # train_inflow_mae.result(),
                        # train_outflow_mae.result())
                    print(template)
                    write_result(result_output_path, template + '\n')
                    with summary_writer.as_default():
                        tf.summary.scalar("train_inflow_rmse", train_inflow_rmse.result(), step=epoch + 1)
                        tf.summary.scalar("train_outflow_rmse", train_outflow_rmse.result(), step=epoch + 1)

                eval_rmse = (train_inflow_rmse.result() + train_outflow_rmse.result()) / 2

                if check_flag == False and earlystop_helper.refresh_status(eval_rmse):
                    check_flag = True

                if check_flag:
                    print(
                        "Validation Result (after Min-Max Normalization, filtering out grids with flow less than consideration threshold): ")
                    evaluate(val_dataset, epoch, testing=False)
                    with summary_writer.as_default():
                        tf.summary.scalar("test_inflow_rmse", test_inflow_rmse.result(), step=epoch + 1)
                        tf.summary.scalar("test_outflow_rmse", test_outflow_rmse.result(), step=epoch + 1)
                        # tf.summary.scalar("test_inflow_mae", test_inflow_mae.result(), step=epoch + 1)
                        # tf.summary.scalar("test_outflow_mae", test_outflow_mae.result(), step=epoch + 1)
                    earlystop_flag = earlystop_helper.check(test_inflow_rmse.result() + test_outflow_rmse.result(), epoch)

                ckpt_save_path = ckpt_manager.save()
                print('Saving checkpoint for epoch {} at {}\n'.format(epoch + 1, ckpt_save_path))

                if earlystop_flag:
                    print("Early stoping...")
                    ckpt.restore(ckpt_manager.checkpoints[- self.args.earlystop_patience_stsan - 1])
                    print('Checkpoint restored!! At epoch {}\n'.format(
                        int(epoch - self.args.earlystop_patience_stsan)))
                    break

                if reshuffle_helper.check(epoch):
                    train_dataset, val_dataset = self.dataset_generator.load_dataset('train', self.args.load_saved_data,
                                                                                     strategy)

                with summary_writer.as_default():
                    tf.summary.scalar("epoch_time", time.time() - start, step=epoch + 1)
                print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

            print(
                "Start testing (without Min-Max Normalization, filtering out grids with flow less than consideration threshold):")
            write_result(result_output_path,
                         "Start testing (without Min-Max Normalization, filtering out grids with flow less than consideration threshold):\n")
            evaluate(test_dataset, epoch, testing=True)
