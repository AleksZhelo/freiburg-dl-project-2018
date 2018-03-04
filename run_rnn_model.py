from datetime import datetime

import os

import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold

from models.lstm_tf_decov import LSTM_TF_DeCov
from util.common import normalized, ensure_dir, fill_pred_lstm_batch, fill_lstm_batch, date2str

from util.loader import load_data_as_numpy
from util.common import loss as mse


# TODO: track best epoch number for early stopping and retrain on whole data?
def run_rnn_model(session, configs, learning_curves, log_dir,
                  save_dir, model_class, n_input, n_test,
                  normalize, train_epochs, batch_size, eval_every, params,
                  early_stopping=False, patience=50,
                  tf_seed=1123, numpy_seed=1123, verbose=True):
    tf.set_random_seed(tf_seed)

    input_tensor = tf.placeholder(tf.float32, [None, None, 6])
    target = tf.placeholder(tf.float32, [None, None, 1])
    c1 = tf.placeholder(tf.float32, [None, 64])
    h1 = tf.placeholder(tf.float32, [None, 64])
    c2 = tf.placeholder(tf.float32, [None, 64])
    h2 = tf.placeholder(tf.float32, [None, 64])
    initial_state = (tf.nn.rnn_cell.LSTMStateTuple(c1, h1),
                     tf.nn.rnn_cell.LSTMStateTuple(c2, h2))
    phase = tf.placeholder(tf.bool, name='phase')

    rnn = model_class(input_tensor, target, initial_state, phase, **params)

    model_desc = '{0}_{1}_{2}'.format(
        model_class.__name__,
        '_'.join(['{0}={1}'.format(a, b) for a, b in zip(params.keys(), params.values())]),
        date2str(datetime.now())
    )

    if log_dir is not None:
        train_summary_writer = tf.summary.FileWriter(
            os.path.join(log_dir, model_desc),
            session.graph
        )

    x = np.zeros((batch_size, n_input, 6), dtype=np.float32)
    y = np.zeros((batch_size, n_input, 1), dtype=np.float32)

    k_fold = KFold(n_splits=3, shuffle=True, random_state=1)
    saver = tf.train.Saver()
    performances_valid = -np.ones(3)
    performances_test = np.zeros(3)

    current_fold = 0
    rs_ = np.random.RandomState(numpy_seed)
    for train_indices, test_indices in k_fold.split(configs):
        if log_dir is not None:
            t_loss_summary = tf.summary.scalar('losses/TrainingLoss_fold:{0}'.format(current_fold),
                                               rnn.loss)
            v_loss_summary = tf.summary.scalar('losses/ValidationLoss_fold:{0}'.format(current_fold),
                                               rnn.loss_pure)

        session.run(tf.global_variables_initializer())
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        # not counting the validation set being taken out to keep this constant with/without ES
        num_train_samples = train_indices.shape[0]
        epoch_steps = int(np.ceil(num_train_samples / batch_size))

        if early_stopping:
            rs_.shuffle(train_indices)
            valid_length = int(np.ceil(train_indices.shape[0] * 0.2))
            valid_indices = train_indices[:valid_length]
            train_indices = train_indices[valid_length:]

            train_configs = configs[train_indices]
            train_curves = learning_curves[train_indices]
            valid_configs = configs[valid_indices]
            valid_curves = learning_curves[valid_indices]

            best_valid = float('inf')
            counter = 0
        else:
            train_configs = configs[train_indices]
            train_curves = learning_curves[train_indices]
            valid_configs = None
            valid_curves = None

        test_configs = configs[test_indices]
        test_curves = learning_curves[test_indices]

        if normalize:
            train_configs, mean, std = normalized(train_configs)
            test_configs, _, _ = normalized(test_configs, mean, std)
            if early_stopping:
                valid_configs, _, _ = normalized(valid_configs, mean, std)

        total_epochs = 0
        curr_steps = 0
        while total_epochs < train_epochs:
            loss_data = np.zeros(epoch_steps)
            for step_ in range(epoch_steps):
                x, y = fill_lstm_batch(x, y, n_input, train_configs, train_curves, rs_)
                loss, _, _, state = session.run(
                    [rnn.loss, rnn.optimize, update_ops, rnn.lstm_final_state],
                    {rnn.input_tensor: x, rnn.target: y, phase: 1,
                     c1: np.zeros((batch_size, 64), dtype=np.float32),
                     h1: np.zeros((batch_size, 64), dtype=np.float32),
                     c2: np.zeros((batch_size, 64), dtype=np.float32),
                     h2: np.zeros((batch_size, 64), dtype=np.float32)}
                )
                loss_data[step_] = loss
                curr_steps += 1

            total_epochs += 1

            if total_epochs % eval_every == 0:
                pred_full = predict_curve(initial_state, n_test, 40,
                                          phase, rnn, session,
                                          test_configs, test_curves)
                error_last_pred = pred_full[:, -1].reshape(-1)
                test_mse = mse(error_last_pred, test_curves[:, -1])
                if early_stopping:
                    pred_full_valid = predict_curve(initial_state, n_test, 40,
                                                    phase, rnn, session,
                                                    valid_configs, valid_curves)
                    error_last_pred_valid = pred_full_valid[:, -1].reshape(-1)
                    valid_mse = mse(error_last_pred_valid, valid_curves[:, -1])
                    print('Epoch {0} test loss: {1:.5f}, valid loss: {3:.5f} train_loss: {2:.5f}'.format(
                        total_epochs,
                        test_mse,
                        loss_data.mean(),
                        valid_mse)
                    )
                    if valid_mse < best_valid:
                        best_valid = valid_mse
                        counter = 0
                        if save_dir is not None:
                            print('saved model')
                            saver.save(session, os.path.join(
                                save_dir,
                                '{0}_fold_{1}.ckpt'.format(model_desc, current_fold)
                            ))
                    else:
                        counter += 1
                        if counter > patience:
                            print('restored model')
                            saver.restore(session, os.path.join(
                                save_dir,
                                '{0}_fold_{1}.ckpt'.format(model_desc, current_fold)
                            ))
                            break
                else:
                    print('Epoch {0} test loss: {1:.5f}, train_loss: {2:.5f}'.format(
                        total_epochs,
                        test_mse,
                        loss_data.mean())
                    )

            if log_dir is not None:  # TODO: broken
                if total_epochs % eval_every == 0:
                    sm, t_loss = session.run([t_loss_summary, rnn.loss],
                                             {rnn.input_tensor: train_configs,
                                              rnn.target: train_curves[:, -1].reshape(-1, 1),
                                              phase: 0})
                    train_summary_writer.add_summary(sm, total_epochs)

                    sm, ev_loss, pure_loss = session.run([v_loss_summary, rnn.loss, rnn.loss_pure],
                                                         {rnn.input_tensor: test_configs,
                                                          rnn.target: test_curves[:, -1].reshape(-1, 1),
                                                          phase: 0})
                    train_summary_writer.add_summary(sm, total_epochs)

        pred_full = predict_curve(initial_state, n_test, 40,
                                  phase, rnn, session,
                                  test_configs, test_curves)
        performances_test[current_fold] = mse(pred_full[:, -1].reshape(-1), test_curves[:, -1])
        if early_stopping:
            performances_valid[current_fold] = best_valid
        # print(session.run(mlp.prediction, {mlp.input_tensor: test_configs}) - test_curves[:, -1].reshape(-1, 1))
        current_fold += 1
    if verbose:
        if early_stopping:
            print('mean cross-validation valid loss: {0}'.format(performances_valid.mean()))
        print('mean cross-validation test loss: {0}, params: {1}'.format(performances_test.mean(), params))
    return performances_test.mean(), performances_valid.mean()


def predict_curve(initial_state, n_input, n_output, phase, rnn, session, configs, curves):
    pred_x = np.zeros((configs.shape[0], n_input, 6), dtype=np.float32)
    pred_x = fill_pred_lstm_batch(pred_x, n_input, configs, curves)

    pred, pred_state \
        = session.run([rnn.prediction,
                       rnn.lstm_final_state],
                      {rnn.input_tensor: pred_x, phase: 0,
                       initial_state[0].c: np.zeros((pred_x.shape[0], 64), dtype=np.float32),
                       initial_state[0].h: np.zeros((pred_x.shape[0], 64), dtype=np.float32),
                       initial_state[1].c: np.zeros((pred_x.shape[0], 64), dtype=np.float32),
                       initial_state[1].h: np.zeros((pred_x.shape[0], 64), dtype=np.float32)})
    pred_next_list = []
    extra_x = np.zeros((pred_x.shape[0], 1, 6))
    extra_x[:, 0, :5] = 0
    extra_x[:, 0, 5] = pred[:, -1, 0]
    for extra_step in range(n_input, n_output):  # TODO: re-check
        pred_next, pred_state = session.run([rnn.prediction, rnn.lstm_final_state],
                                            {rnn.input_tensor: extra_x,
                                             phase: 0,
                                             initial_state[0].c: pred_state[0].c,
                                             initial_state[0].h: pred_state[0].h,
                                             initial_state[1].c: pred_state[1].c,
                                             initial_state[1].h: pred_state[1].h})
        pred_next_list.append(pred_next)
        extra_x[:, 0, 5] = pred_next[:, 0, 0]
    pred_full = np.concatenate(
        (pred, np.concatenate(pred_next_list, axis=1)), axis=1
    )
    return pred_full


if __name__ == '__main__':
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    ensure_dir(log_dir)
    save_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
    ensure_dir(save_dir)
    res_dir = os.path.join(os.path.dirname(__file__), 'optimization_results')
    ensure_dir(res_dir)

    configs, learning_curves = load_data_as_numpy()

    batch_size = 32
    train_epochs = 1500
    eval_every = 1
    normalize = True
    early_stopping = True
    patience = 150

    model = LSTM_TF_DeCov

    with tf.Session() as session:
        params = {
            'learning_rate': 0.001,
            'reg_weight': 0.05,
            'batch_size': batch_size,
            'exponential_decay': False,
            'decay_rate': 0.1,
            'decay_steps': 200 * 176 / batch_size  # hacky
        }
        training_start = date2str(datetime.now())
        # TODO: evaluate MSE@40[:5], MSE@40[:10], MSE@40[:20], MSE@40[:30]
        # TODO: and use their mean for best_valid
        # TODO: also simply report and save
        perf_test, perf_valid = \
            run_rnn_model(session, configs, learning_curves, None, save_dir,
                          model, 20, 30, normalize, train_epochs, batch_size, eval_every, params,
                          early_stopping=early_stopping, patience=patience,
                          tf_seed=1123, numpy_seed=1123, verbose=True)

        with open(os.path.join(res_dir, 'rnn_results.txt'), 'a') as f:
            f.write('{0}, started {1}, finished {2}\n'.format(
                model.__name__, training_start, date2str(datetime.now())
            ))
            f.write('cv_loss_test: {0:.6f}, cv_loss_valid: {1:.6f}\n'.format(
                perf_test, perf_valid,
            ))
            f.write('training params: {0} \nmodel params: {1}\n'.format(
                dict(
                    batch_size=batch_size,
                    train_epochs=train_epochs,
                    eval_every=eval_every,
                    normalize=normalize,
                    early_stopping=early_stopping,
                    patience=patience
                ),
                params
            ))
            f.write('------------------------------------------------------\n')
