import os
import json

import numpy as np

from util.common import ensure_dir, get_pd_frame_task3
from util.plots import scatter, extrapolation


def save_scatter_plot(y, y_hat, loss, name, plot_dir):
    scatter(
        y, y_hat,
        '{0}: MSE {1}'.format(name, loss),
        os.path.join(plot_dir, '{0}_scatter.png'.format(name))
    )


def save_extrapolation_plot(true_curve, extrapolation_list, n_steps, name, plot_dir):
    ensure_dir(plot_dir)
    extrapolation(
        true_curve, extrapolation_list,
        n_steps,
        name,
        os.path.join(plot_dir, '{0}_extrapolation.png'.format(name))
    )


if __name__ == '__main__':
    with open('out/task3_best_models_results.txt', 'r') as f:
        task3_results = json.load(f)

    # filtered_results = list(filter(lambda item: 'DeCov_MLP_init' not in item[0], task3_results))
    filtered_results = list(filter(
        lambda item: 'LSTM_TF_10' == item[0] or 'LSTM_TF_DeCov_10' == item[0] or
                     'LSTM_TF_None' == item[0] or 'LSTM_TF_DeCov_None' == item[0],
        task3_results
    ))

    scatter_plots_dir = os.path.join(os.path.dirname(__file__), 'task3', 'plots', 'scatter')
    ensure_dir(scatter_plots_dir)
    extrapolation_plots_dir = os.path.join(os.path.dirname(__file__), 'task3', 'plots', 'extrapolation')
    extrapolation_100_plots_dir = os.path.join(os.path.dirname(__file__), 'task3', 'plots', 'extrapolation_100')

    var_input_losses = [res[2] for res in filtered_results]
    losses = [np.mean(cv_test) for cv_test in var_input_losses]
    n_input = [res[1] if res[1] else 'rnd' for res in filtered_results]
    estimators = [res[0] for res in filtered_results]
    for i, est in enumerate(estimators):
        idx = len(est) - est[::-1].index('_') - 1
        estimators[i] = est[:idx]

    frame = get_pd_frame_task3(losses, var_input_losses,
                               n_input, None, estimators)
    frame = frame.sort_values(['n_train', 'loss_mean'], ascending=[False, True])
    # frame.to_latex('task3_table.tex')
    print(frame)

    n_test = [5, 10, 20, 30]
    n_test_idx = 1  # corresponds to predicting from 10 points
    for i, res in enumerate(filtered_results):
        curve = np.concatenate([np.array(res[3][j][0]) for j in range(3)])
        curve_hat = [np.concatenate([np.array(res[3][j][1][k]) for j in range(3)]).squeeze()
                     for k in range(len(n_test))]
        curve_100_hat = np.concatenate([np.array(res[3][j][2][0]) for j in range(3)]).squeeze()

        idx = curve.sum(axis=1).argsort()
        curve = curve[idx]
        curve_hat = [c_h[idx] for c_h in curve_hat]
        curve_100_hat = curve_100_hat[idx]

        errors_10 = curve[:, -1] - curve_hat[1][:, -1]
        np.savetxt('{0}_{1}_errors.txt'.format(estimators[i], n_input[i]), errors_10)

        # save_scatter_plot(curve[:, -1], curve_hat[n_test_idx][:, -1], np.round(res[2][n_test_idx], 6),
        #                   '{0}_{1}_from_{2}'.format(estimators[i], n_input[i], n_test[n_test_idx]),
        #                   scatter_plots_dir)
        #
        # for c_num in [0, 10, 100, 200, 250]:
        #     save_extrapolation_plot(curve[0], [(str(n_test[j]), curve_hat[j][0][n_test[j]:]) for j in range(len(n_test))],
        # save_extrapolation_plot(curve[c_num], [(str(n_test[j]), curve_hat[j][c_num]) for j in range(len(n_test))],
        #                         40, '{0}_{1} curve {2:03}'.format(estimators[i], n_input[i], c_num),
        #                         os.path.join(extrapolation_plots_dir, str(c_num)))
        #
        # save_extrapolation_plot(curve[c_num], [('40', curve_100_hat[c_num])],
        #                         100, '{0}_{1} curve {2:03} pred_100'.format(estimators[i], n_input[i], c_num),
        #                         os.path.join(extrapolation_100_plots_dir, str(c_num)))
