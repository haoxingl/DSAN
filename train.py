from __future__ import absolute_import, division, print_function, unicode_literals

import time, os, codecs, json

import numpy as np
from utils.tools import DatasetGenerator, write_result, create_masks
from utils.CustomSchedule import CustomSchedule
from utils.EarlystopHelper import EarlystopHelper
from utils.Metrics import MAE, MAPE
from models import DSAN

import tensorflow as tf

from data_parameters import data_parameters


class TrainModel:
    def __init__(self, model_index, args):

        """ use mirrored strategy for distributed training """
        self.strategy = tf.distribute.MirroredStrategy()
        strategy = self.strategy
        print('Number of GPU devices: {}'.format(strategy.num_replicas_in_sync))

        param = data_parameters[args.dataset]
        self.param = param

        self.model_index = model_index
        if args.test_model:
            args.n_layer = 1
            args.d_model = 8
            args.dff = 32
            args.n_head = 1
            args.conv_layer = 1
            args.conv_filter = 8
            args.n_w = 0
            args.n_d = 1
            args.n_wd_times = 1
            args.n_p = 0
            args.n_before = 0
            args.n_pred = 3
            args.l_half = 1
            args.l_half_g = 2
        self.args = args
        self.args.l_hist = (args.n_w + args.n_d) * args.n_wd_times + args.n_p
        self.GLOBAL_BATCH_SIZE = args.BATCH_SIZE * strategy.num_replicas_in_sync
        self.dataset_generator = DatasetGenerator(args.d_model,
                                                  args.dataset,
                                                  self.GLOBAL_BATCH_SIZE,
                                                  args.n_w,
                                                  args.n_d,
                                                  args.n_wd_times,
                                                  args.n_p,
                                                  args.n_before,
                                                  args.n_pred,
                                                  args.l_half,
                                                  args.l_half_g,
                                                  args.pre_shuffle,
                                                  args.same_padding,
                                                  args.test_model)

        self.es_patiences = [5, args.es_patience]
        self.es_threshold = args.es_threshold
        self.data_max = param['data_max'][:param['pred_type']]

        self.test_threshold = [param['test_threshold'][i] / self.data_max[i] for i in range(param['pred_type'])]

    def train(self):
        strategy = self.strategy
        args = self.args
        param = self.param
        n_pred = args.n_pred
        pred_type = param['pred_type']
        data_name = param['data_name']
        weights = args.weights
        test_model = args.test_model

        result_output_path = "results/dsan/{}.txt".format(self.model_index)

        train_dataset = self.dataset_generator.build_dataset('train', args.load_saved_data,
                                                             strategy, args.st_revert, args.no_save)
        test_dataset = None

        with strategy.scope():

            def tf_summary_scalar(summary_writer, name, value, step):
                with summary_writer.as_default():
                    tf.summary.scalar(name, value, step=step)

            def print_verbose(epoch, final_test):
                if final_test:
                    template_rmse = "RMSE:\n"
                    for i in range(pred_type):
                        template_rmse += '{}:'.format(data_name[i])
                        for j in range(n_pred):
                            template_rmse += ' {}. {:.2f}({:.6f})'.format(
                                j + 1,
                                rmse_test[j][i].result() * self.data_max[i],
                                rmse_test[j][i].result()
                            )
                        template_rmse += '\n'
                    template_mae = "MAE:\n"
                    for i in range(pred_type):
                        template_mae += '{}:'.format(data_name[i])
                        for j in range(n_pred):
                            template_mae += ' {}. {:.2f}({:.6f})'.format(
                                j + 1,
                                mae_test[j][i].result() * self.data_max[i],
                                mae_test[j][i].result()
                            )
                        template_mae += '\n'
                    template_mape = "MAPE:\n"
                    for i in range(pred_type):
                        template_mape += '{}:'.format(data_name[i])
                        for j in range(n_pred):
                            template_mape += ' {}. {:.2f}'.format(j + 1, mape_test[j][i].result())
                        template_mape += '\n'
                    template = "Final:\n" + template_rmse + template_mae + template_mape
                    write_result(result_output_path, template)
                else:
                    template = "Epoch {} RMSE:\n".format(epoch + 1)
                    for i in range(pred_type):
                        template += '{}:'.format(data_name[i])
                        for j in range(n_pred):
                            template += " {}. {:.6f}".format(j + 1, rmse_test[j][i].result())
                        template += "\n"
                    write_result(result_output_path,
                                 'Validation Result (Min-Max Norm, filtering out trivial grids):\n' + template)

            loss_object = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

            def loss_function(real, pred):
                loss_ = loss_object(real, pred)
                return tf.nn.compute_average_loss(loss_, global_batch_size=self.GLOBAL_BATCH_SIZE)

            rmse_train = [tf.keras.metrics.RootMeanSquaredError(dtype=tf.float32) for _ in range(pred_type)]
            rmse_test = [[tf.keras.metrics.RootMeanSquaredError(dtype=tf.float32)
                          for _ in range(pred_type)] for _ in range(n_pred)]

            mae_test = [[MAE() for _ in range(pred_type)] for _ in range(n_pred)]
            mape_test = [[MAPE() for _ in range(pred_type)] for _ in range(n_pred)]

            learning_rate = CustomSchedule(args.d_model, args.warmup_steps)

            optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

            dsan = DSAN(args.n_layer,
                        args.d_model,
                        args.n_head,
                        args.dff,
                        args.conv_layer,
                        args.conv_filter,
                        args.l_hist,
                        args.r_d)

            def train_step(dae_inp_g, dae_inp, dae_inp_ex, sad_inp, sad_inp_ex, cors, cors_g, y):

                padding_mask_g, padding_mask, combined_mask = \
                    create_masks(dae_inp_g[..., :pred_type], dae_inp[..., :pred_type], sad_inp)

                with tf.GradientTape() as tape:
                    predictions, _, _ = dsan(dae_inp_g, dae_inp, dae_inp_ex, sad_inp, sad_inp_ex, cors, cors_g, True,
                                             padding_mask, padding_mask_g, combined_mask)
                    if type(weights) is np.ndarray:
                        loss = loss_function(y * weights, predictions * weights)
                    else:
                        loss = loss_function(y, predictions)

                gradients = tape.gradient(loss, dsan.trainable_variables)
                optimizer.apply_gradients(zip(gradients, dsan.trainable_variables))

                for i in range(pred_type):
                    rmse_train[i](y[..., i], predictions[..., i])

                return loss

            @tf.function
            def distributed_train_step(dae_inp_g, dae_inp, dae_inp_ex, sad_inp, sad_inp_ex, cors, cors_g, y):
                per_replica_losses = strategy.experimental_run_v2 \
                    (train_step, args=(dae_inp_g, dae_inp, dae_inp_ex, sad_inp, sad_inp_ex, cors, cors_g, y,))

                return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

            def test_step(dae_inp_g, dae_inp, dae_inp_ex, sad_inp, sad_inp_ex, cors, cors_g, y, final_test=False):
                targets = sad_inp[:, :1, :]
                for i in range(n_pred):
                    tar_inp_ex = sad_inp_ex[:, :i + 1, :]
                    padding_mask_g, padding_mask, combined_mask = \
                        create_masks(dae_inp_g[..., :pred_type], dae_inp[..., :pred_type], targets)

                    predictions, _, _ = dsan(dae_inp_g, dae_inp, dae_inp_ex, targets, tar_inp_ex, cors, cors_g, False,
                                             padding_mask, padding_mask_g, combined_mask)

                    """ here we filter out all nodes where their real flows are less than 10 """
                    for j in range(pred_type):
                        real = y[:, i, j] * (weights[i, j] if type(weights) is np.ndarray else 1)
                        pred = predictions[:, -1, j] * (weights[i, j] if type(weights) is np.ndarray else 1)
                        mask = tf.where(tf.math.greater(real, self.test_threshold[j]))
                        masked_real = tf.gather_nd(real, mask)
                        masked_pred = tf.gather_nd(pred, mask)
                        rmse_test[i][j](masked_real, masked_pred)
                        if final_test:
                            mae_test[i][j](masked_real, masked_pred)
                            mape_test[i][j](masked_real, masked_pred)

                    targets = tf.concat([targets, predictions[:, -1:, :]], axis=-2)

            @tf.function
            def distributed_test_step(dae_inp_g, dae_inp, dae_inp_ex, sad_inp, sad_inp_ex, cors, cors_g, y, final_test):
                return strategy.experimental_run_v2(test_step, args=(
                    dae_inp_g, dae_inp, dae_inp_ex, sad_inp, sad_inp_ex, cors, cors_g, y, final_test,))

            def evaluate(eval_dataset, epoch, verbose=1, final_test=False):
                for i in range(n_pred):
                    for j in range(pred_type):
                        rmse_test[i][j].reset_states()

                for (batch, (inp, tar)) in enumerate(eval_dataset):
                    dae_inp_g = inp["dae_inp_g"]
                    dae_inp = inp["dae_inp"]
                    dae_inp_ex = inp["dae_inp_ex"]
                    sad_inp = inp["sad_inp"]
                    sad_inp_ex = inp["sad_inp_ex"]
                    cors = inp["cors"]
                    cors_g = inp["cors_g"]

                    y = tar["y"]

                    distributed_test_step(
                        dae_inp_g, dae_inp, dae_inp_ex, sad_inp, sad_inp_ex, cors, cors_g, y, final_test)

                if verbose:
                    print_verbose(epoch, final_test)

            """ Start training... """
            es_flag = False
            check_flag = False
            es_helper = EarlystopHelper(self.es_patiences, self.es_threshold)
            summary_writer = tf.summary.create_file_writer(os.environ['HOME'] + '/tensorboard/dsan/{}'.format(self.model_index))
            step_cnt = 0
            last_epoch = 0

            checkpoint_path = "./checkpoints/dsan/{}".format(self.model_index)

            ckpt = tf.train.Checkpoint(DSAN=dsan, optimizer=optimizer)

            ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path,
                                                      max_to_keep=(args.es_patience + 1))

            if os.path.isfile(checkpoint_path + '/ckpt_record.json'):
                with codecs.open(checkpoint_path + '/ckpt_record.json', encoding='utf-8') as json_file:
                    ckpt_record = json.load(json_file)

                last_epoch = ckpt_record['epoch']
                es_flag = ckpt_record['es_flag']
                check_flag = ckpt_record['check_flag']
                es_helper.load_ckpt(checkpoint_path)
                step_cnt = ckpt_record['step_cnt']

                ckpt.restore(ckpt_manager.checkpoints[-1])
                write_result(result_output_path, "Check point restored at epoch {}".format(last_epoch))

            write_result(result_output_path, "Start training...\n")

            for epoch in range(last_epoch, args.MAX_EPOCH + 1):

                if es_flag or epoch == args.MAX_EPOCH:
                    print("Early stoping...")
                    if es_flag:
                        ckpt.restore(ckpt_manager.checkpoints[0])
                    else:
                        ckpt.restore(ckpt_manager.checkpoints[es_helper.get_bestepoch() - epoch - 1])
                    print('Checkpoint restored!! At epoch {}\n'.format(es_helper.get_bestepoch()))
                    break

                start = time.time()

                for i in range(pred_type):
                    rmse_train[i].reset_states()

                for (batch, (inp, tar)) in enumerate(train_dataset):

                    dae_inp_g = inp["dae_inp_g"]
                    dae_inp = inp["dae_inp"]
                    dae_inp_ex = inp["dae_inp_ex"]
                    sad_inp = inp["sad_inp"]
                    sad_inp_ex = inp["sad_inp_ex"]
                    cors = inp["cors"]
                    cors_g = inp["cors_g"]

                    y = tar["y"]

                    total_loss = distributed_train_step(
                        dae_inp_g, dae_inp, dae_inp_ex, sad_inp, sad_inp_ex, cors, cors_g, y)

                    step_cnt += 1
                    tf_summary_scalar(summary_writer, "total_loss", total_loss, step_cnt)

                    if (batch + 1) % 100 == 0 and args.verbose_train:
                        template = 'Epoch {} Batch {} RMSE:'.format(epoch + 1, batch + 1)
                        for i in range(pred_type):
                            template += ' {} {:.6f}'.format(data_name[i], rmse_train[i].result())

                if args.verbose_train:
                    template = ''
                    for i in range(pred_type):
                        template += ' {} {:.6f}'.format(data_name[i], rmse_train[i].result())
                        tf_summary_scalar(
                            summary_writer, "rmse_train_{}".format(data_name[i]), rmse_train[i].result(), epoch + 1)
                    template = 'Epoch {}{}\n'.format(epoch + 1, template)
                    write_result(result_output_path, template)

                eval_rmse = 0.0
                for i in range(pred_type):
                    eval_rmse += float(rmse_train[i].result().numpy())

                if test_model or (not check_flag and es_helper.refresh_status(eval_rmse)):
                    check_flag = True
                    val_dataset = self.dataset_generator.build_dataset(
                        'val', args.load_saved_data, strategy, args.st_revert, args.no_save)

                if check_flag:
                    evaluate(val_dataset, epoch, final_test=False)
                    es_rmse = [0.0 for _ in range(pred_type)]
                    for i in range(pred_type):
                        for j in range(n_pred):
                            if type(weights) is np.ndarray:
                                es_rmse[i] += float(rmse_test[j][i].result().numpy() * weights[j, i] / n_pred)
                            else:
                                es_rmse[i] += float(rmse_test[j][i].result().numpy() / n_pred)
                        tf_summary_scalar(summary_writer, "rmse_test_{}".format(data_name[i]), es_rmse[i], epoch + 1)
                    es_flag = es_helper.check(es_rmse[0] + es_rmse[1], epoch)
                    tf_summary_scalar(summary_writer, "best_epoch", es_helper.get_bestepoch(), epoch + 1)
                    if args.always_test and (epoch + 1) % args.always_test == 0:
                        if not test_dataset:
                            test_dataset = self.dataset_generator.build_dataset(
                                'test', args.load_saved_data, strategy, args.st_revert, args.no_save)
                        write_result(result_output_path, "Always Test:")
                        evaluate(test_dataset, epoch)

                ckpt_save_path = ckpt_manager.save()
                ckpt_record = {'epoch': epoch + 1, 'best_epoch': es_helper.get_bestepoch(),
                               'check_flag': check_flag, 'es_flag': es_flag, 'step_cnt': step_cnt}
                ckpt_record = json.dumps(ckpt_record, indent=4)
                with codecs.open(checkpoint_path + '/ckpt_record.json', 'w', 'utf-8') as outfile:
                    outfile.write(ckpt_record)
                es_helper.save_ckpt(checkpoint_path)
                print('Save checkpoint for epoch {} at {}\n'.format(epoch + 1, ckpt_save_path))

                tf_summary_scalar(summary_writer, "epoch_time", time.time() - start, epoch + 1)
                print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

                if test_model:
                    es_flag = True

            write_result(result_output_path, "Start testing (filtering out trivial grids):")
            test_dataset = self.dataset_generator.build_dataset(
                'test', args.load_saved_data, strategy, args.st_revert, args.no_save)
            evaluate(test_dataset, epoch, final_test=True)
            for i in range(pred_type):
                tf_summary_scalar(summary_writer, "final_rmse_{}".format(data_name[i]), rmse_test[0][i].result(), 1)
