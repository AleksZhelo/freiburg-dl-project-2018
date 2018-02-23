from __future__ import print_function

import numpy as np
import math


# TODO: comment cryptic variable names
# TODO: hyperband is not very good at optimizing the learning rate
# http://fastml.com/tuning-hyperparams-fast-with-hyperband/
class Hyperband(object):

    def __init__(self, sample_params_function, run_model_function,
                 max_epochs=300, reduction_factor=3):
        self.sample_params = sample_params_function
        self.run_model = run_model_function

        self.R = max_epochs  # max resource limit
        self.eta = reduction_factor
        self.s_max = int(math.floor(math.log(self.R, self.eta)))
        self.B = (self.s_max + 1) * self.R

    def run(self, verbose=False):
        def log(*args, **kwargs):
            if verbose:
                print(*args, **kwargs)

        runs_total = 0
        results_total = []
        for s in range(self.s_max, -1, -1):
            log("-------- s: {0} --------".format(s))
            n = int(math.ceil(self.B / self.R * self.eta ** s / (s + 1)))
            r = self.R * self.eta ** (-s)

            log("n: {0}, r: {1}".format(n, r))

            T = np.array([self.sample_params() for _ in range(n)])
            for i in range(0, s + 1):
                n_i = math.ceil(n * self.eta ** (-i))
                r_i = r * self.eta ** i
                runs_total += n_i
                log("n_{2}: {0}, r_{2}: {1}".format(n_i, r_i, i))
                L = np.array([self.run_model(params, r_i) for params in T])
                idx = np.argsort(L)
                T = T[idx[:int(math.ceil(n_i / self.eta))]]
                log("keeping: {0}".format(math.ceil(n_i / self.eta)))
                L = L[idx]
                for config, loss in zip(T, L):
                    results_total.append(dict(loss=loss, epochs=r_i, config=config))

        log("runs total: {0}".format(runs_total))
        return results_total


if __name__ == "__main__":
    hyperband = Hyperband(None, None)
    hyperband.run(verbose=True)
