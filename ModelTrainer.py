from __future__ import absolute_import, division, print_function, unicode_literals

import time, os, codecs, json

from utils.tools import DatasetGenerator, write_result, create_masks
from utils.CustomSchedule import CustomSchedule
from utils.EarlystopHelper import EarlystopHelper
from utils.Metrics import MAE, MAPE
from models import STSAN_XL

import tensorflow as tf

import parameters_nyctaxi
import parameters_nycbike
import parameters_ctm


class ModelTrainer:
    def __init__(self, model_index, args):

        gpus = tf.config.experimental.list_physical_devices('GPU')

        if gpus and args.gm_growth:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)

        """ use mirrored strategy for distributed training """
        self.strategy = tf.distribute.MirroredStrategy()
        strategy = self.strategy
        print('Number of GPU devices: {}'.format(strategy.num_replicas_in_sync))

        param = None
        param = parameters_nyctaxi if args.dataset == 'taxi' else param
        param = parameters_nycbike if args.dataset == 'bike' else param
        param = parameters_ctm if args.dataset == 'ctm' else param
        self.param = param

        self.model_index = model_index
        self.args = args
        if args.test_model:
            args.num_layers = 1
            args.d_model = 8
            args.dff = 32
            args.num_heads = 1
            args.cnn_layers = 1
            args.cnn_filters = 8
            args.n_hist_week = 0
            args.n_hist_day = 1
            args.n_hist_int = 1
            args.n_curr_int = 0
            args.n_int_before = 0
            args.n_pred = 3
            args.local_block_len = 1
            args.local_block_len_g = 2

        self.args.seq_len = (args.n_hist_week + args.n_hist_day) * args.n_hist_int + args.n_curr_int
        if args.weight_1:
            self.args.weight_2 = 1 - args.weight_1
        else:
            self.args.weight_2 = None
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
                                                  args.local_block_len,
                                                  args.local_block_len_g,
                                                  args.test_model)

        self.es_patiences = [5, args.es_patience]
        self.es_threshold = args.es_threshold
        if args.dataset in ['taxi', 'bike']:
            self.data_max = [self.param.data_max, self.param.data_max]

        else:
            self.data_max = param.data_max

        self.test_threshold = [param.test_threshold[0] / self.data_max[0], param.test_threshold[0] / self.data_max[1]]

    def train(self):
        strategy = self.strategy
        args = self.args
        param = self.param
        type_pred = param.data_type
        test_model = args.test_model

        result_output_path = "results/stsan_xl/{}.txt".format(self.model_index)

        train_dataset = self.dataset_generator.build_dataset('train', args.load_saved_data,
                                                             strategy, args.st_revert, args.no_save)
        val_dataset = self.dataset_generator.build_dataset('val', args.load_saved_data,
                                                           strategy, args.st_revert, args.no_save)
        test_dataset = self.dataset_generator.build_dataset('test', args.load_saved_data, strategy,
                                                            args.st_revert, args.no_save)

        with strategy.scope():

            def tf_summary_scalar(summary_writer, name, value, step):
                with summary_writer.as_default():
                    tf.summary.scalar(name, value, step=step)

            def print_verbose(epoch, final_test):
                if final_test:
                    template_rmse = "RMSE:\n"
                    for i in range(type_pred):
                        template_rmse += '{}:'.format(param.data_name[i])
                        for j in range(args.n_pred):
                            template_rmse += ' {}. {:.2f}({:.6f})'.format(
                                j + 1,
                                rmse_test[i][j].result() * self.data_max[i],
                                rmse_test[i][j].result()
                            )
                        template_rmse += '\n'
                    template_mae = "MAE:\n"
                    for i in range(type_pred):
                        template_mae += '{}:'.format(param.data_name[i])
                        for j in range(args.n_pred):
                            template_mae += ' {}. {:.2f}({:.6f})'.format(
                                j + 1,
                                mae_test[i][j].result() * self.data_max[i],
                                mae_test[i][j].result()
                            )
                        template_mae += '\n'
                    template_mape = "MAPE:\n"
                    for i in range(type_pred):
                        template_mape += '{}:'.format(param.data_name[i])
                        for j in range(args.n_pred):
                            template_mape += ' {}. {:.2f}'.format(
                                j + 1,
                                mape_test[i][j].result()
                            )
                        template_mape += '\n'
                    template = "Final:\n" + template_rmse + template_mae + template_mape
                    write_result(result_output_path, template)
                else:
                    template = "Epoch {} RMSE:\n".format(epoch + 1)
                    for i in range(type_pred):
                        template += '{}:'.format(param.data_name[i])
                        for j in range(args.n_pred):
                            template += " {}. {:.6f}".format \
                                (j + 1, rmse_test[i][j].result())
                        template += "\n"
                    write_result(result_output_path,
                                 'Validation Result (Min-Max Norm, filtering out trivial grids):\n' + template)

            loss_object = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

            def loss_function(real, pred):
                loss_ = loss_object(real, pred)
                return tf.nn.compute_average_loss(loss_, global_batch_size=self.GLOBAL_BATCH_SIZE)

            rmse_train = [tf.keras.metrics.RootMeanSquaredError(dtype=tf.float32) for _ in range(type_pred)]
            rmse_test = [[tf.keras.metrics.RootMeanSquaredError(dtype=tf.float32) for _ in range(args.n_pred)] for _ in
                         range(type_pred)]

            mae_test = [[MAE() for _ in range(args.n_pred)] for _ in range(type_pred)]
            mape_test = [[MAPE() for _ in range(args.n_pred)] for _ in range(type_pred)]

            learning_rate = CustomSchedule(args.d_model, args.warmup_steps)

            optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

            stsan_xl = STSAN_XL(args.num_layers,
                                args.d_model,
                                args.num_heads,
                                args.dff,
                                args.cnn_layers,
                                args.cnn_filters,
                                args.seq_len,
                                args.dropout_rate)

            def train_step(inp_g, inp_l, inp_ex, dec_inp, dec_inp_ex, cors, cors_g, y):

                padding_mask_g, padding_mask, combined_mask = \
                    create_masks(inp_g[..., :type_pred], inp_l[..., :type_pred], dec_inp)

                with tf.GradientTape() as tape:
                    predictions, _, _ = stsan_xl(inp_g, inp_l, inp_ex, dec_inp, dec_inp_ex, cors, cors_g, True,
                                                 padding_mask, padding_mask_g, combined_mask)
                    if not args.weight_1:
                        loss = loss_function(y, predictions)
                    else:
                        loss = loss_function(y[:, :1, :], predictions[:, :1, :]) * args.weight_1 + \
                               loss_function(y[:, 1:, :], predictions[:, 1:, :]) * args.weight_2

                gradients = tape.gradient(loss, stsan_xl.trainable_variables)
                optimizer.apply_gradients(zip(gradients, stsan_xl.trainable_variables))

                for i in range(type_pred):
                    rmse_train[i](y[..., i], predictions[..., i])

                return loss

            @tf.function
            def distributed_train_step(inp_g, inp_l, inp_ex, dec_inp, dec_inp_ex, cors, cors_g, y):
                per_replica_losses = strategy.experimental_run_v2 \
                    (train_step, args=(inp_g, inp_l, inp_ex, dec_inp, dec_inp_ex, cors, cors_g, y,))

                return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

            def test_step(inp_g, inp_l, inp_ex, dec_inp, dec_inp_ex, cors, cors_g, y, final_test=False):
                targets = dec_inp[:, :1, :]
                for i in range(args.n_pred):
                    tar_inp_ex = dec_inp_ex[:, :i + 1, :]
                    padding_mask_g, padding_mask, combined_mask = \
                        create_masks(inp_g[..., :type_pred], inp_l[..., :type_pred], targets)

                    predictions, _, _ = stsan_xl(inp_g, inp_l, inp_ex, targets, tar_inp_ex, cors, cors_g, False,
                                                 padding_mask, padding_mask_g, combined_mask)

                    """ here we filter out all nodes where their real flows are less than 10 """
                    for j in range(type_pred):
                        real = y[:, i, j]
                        pred = predictions[:, -1, j]
                        mask = tf.where(tf.math.greater(real, self.test_threshold[j]))
                        masked_real = tf.gather_nd(real, mask)
                        masked_pred = tf.gather_nd(pred, mask)
                        rmse_test[j][i](masked_real, masked_pred)
                        if final_test:
                            mae_test[j][i](masked_real, masked_pred)
                            mape_test[j][i](masked_real, masked_pred)

                    targets = tf.concat([targets, predictions[:, -1:, :]], axis=-2)

            @tf.function
            def distributed_test_step(inp_g, inp_l, inp_ex, dec_inp, dec_inp_ex, cors, cors_g, y, final_test):
                return strategy.experimental_run_v2(test_step, args=(
                    inp_g, inp_l, inp_ex, dec_inp, dec_inp_ex, cors, cors_g, y, final_test,))

            def evaluate(eval_dataset, epoch, verbose=1, final_test=False):
                for i in range(args.n_pred):
                    for j in range(type_pred):
                        rmse_test[j][i].reset_states()

                for (batch, (inp, tar)) in enumerate(eval_dataset):
                    inp_g = inp["inp_g"]
                    inp_l = inp["inp_l"]
                    inp_ex = inp["inp_ex"]
                    dec_inp = inp["dec_inp"]
                    dec_inp_ex = inp["dec_inp_ex"]
                    cors = inp["cors"]
                    cors_g = inp["cors_g"]

                    y = tar["y"]

                    distributed_test_step(inp_g, inp_l, inp_ex, dec_inp, dec_inp_ex, cors, cors_g, y, final_test)

                if verbose:
                    print_verbose(epoch, final_test)

            """ Start training... """
            es_flag = False
            check_flag = False
            es_helper = EarlystopHelper(self.es_patiences, self.es_threshold)
            summary_writer = tf.summary.create_file_writer('/home/lxx/tensorboard/stsan_xl/{}'.format(self.model_index))
            step_cnt = 0
            last_epoch = 0

            checkpoint_path = "./checkpoints/stsan_xl/{}".format(self.model_index)

            ckpt = tf.train.Checkpoint(STSAN_XL=stsan_xl, optimizer=optimizer)

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

                for i in range(type_pred):
                    rmse_train[i].reset_states()

                for (batch, (inp, tar)) in enumerate(train_dataset):

                    inp_g = inp["inp_g"]
                    inp_l = inp["inp_l"]
                    inp_ex = inp["inp_ex"]
                    dec_inp = inp["dec_inp"]
                    dec_inp_ex = inp["dec_inp_ex"]
                    cors = inp["cors"]
                    cors_g = inp["cors_g"]

                    y = tar["y"]

                    if args.trace_graph:
                        tf.summary.trace_on(graph=True, profiler=True)
                    total_loss = distributed_train_step(inp_g, inp_l, inp_ex, dec_inp, dec_inp_ex, cors, cors_g, y)
                    if args.trace_graph:
                        with summary_writer.as_default():
                            tf.summary.trace_export(
                                name="stsan_xl_trace",
                                step=step_cnt,
                                profiler_outdir='/home/lxx/tensorboard/stsan_xl/{}'.format(self.model_index))

                    step_cnt += 1
                    tf_summary_scalar(summary_writer, "total_loss", total_loss, step_cnt)

                    if (batch + 1) % 100 == 0 and args.verbose_train:
                        template = 'Epoch {} Batch {} RMSE:'.format(epoch + 1, batch + 1)
                        for i in range(type_pred):
                            template += ' {} {:.6f}'.format(param.data_name[i], rmse_train[i].result())

                if args.verbose_train:
                    template = ''
                    for i in range(type_pred):
                        template += ' {} {:.6f}'.format(param.data_name[i], rmse_train[i].result())
                        tf_summary_scalar(summary_writer, "rmse_train_{}".format(param.data_name[i]), rmse_train[i].result(),
                                          epoch + 1)
                    template = 'Epoch {}{}\n'.format \
                        (epoch + 1, template)
                    write_result(result_output_path, template)

                eval_rmse = 0.0
                for i in range(type_pred):
                    eval_rmse += float(rmse_train[i].result().numpy())
                eval_rmse /= type_pred

                if not check_flag and es_helper.refresh_status(eval_rmse):
                    check_flag = True

                if test_model or check_flag:
                    evaluate(val_dataset, epoch, final_test=False)
                    es_rmse = 0.0
                    for i in range(type_pred):
                        tf_summary_scalar(summary_writer, "rmse_test_{}".format(param.data_name[i]), rmse_test[i][0].result(),
                                          epoch + 1)
                        es_rmse += float(rmse_test[i][0].result().numpy())
                    es_flag = es_helper.check(es_rmse / type_pred, epoch)
                    tf_summary_scalar(summary_writer, "best_epoch", es_helper.get_bestepoch(), epoch + 1)
                    if args.always_test and (epoch + 1) % args.always_test == 0:
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
            evaluate(test_dataset, epoch, final_test=True)
            for i in range(type_pred):
                tf_summary_scalar(summary_writer, "final_rmse_{}".format(param.data_name[i]), rmse_test[i][0].result(), 1)
