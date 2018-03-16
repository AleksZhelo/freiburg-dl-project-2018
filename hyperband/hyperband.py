from __future__ import print_function

import numpy as np
import math


# TODO: comment cryptic variable names
# TODO: hyperband is not very good at optimizing the learning rate
# http://fastml.com/tuning-hyperparams-fast-with-hyperband/
class Hyperband(object):

    def __init__(self, sample_params_function, run_model_function,
                 max_epochs=300, reduction_factor=3, min_r=10):
        self.sample_params = sample_params_function
        self.run_model = run_model_function

        self.R = max_epochs  # max resource limit
        self.eta = reduction_factor
        self.min_r = min_r
        self.s_max = int(math.floor(math.log(self.R, self.eta)))
        self.B = (self.s_max + 1) * self.R

    def run(self, early_stopping=False, verbose=False, dry_run=False):
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

            T = np.array([self.sample_params() for _ in range(n)]) if not dry_run else None
            for i in range(0, s + 1):
                n_i = math.ceil(n * self.eta ** (-i))
                r_i = r * self.eta ** i
                if r_i < self.min_r:
                    r_i += self.min_r
                runs_total += n_i
                log("n_{2}: {0}, r_{2}: {1}".format(n_i, r_i, i))
                runs = [self.run_model(params, r_i) for params in T] if not dry_run else None

                if early_stopping:
                    L = np.array([run[0] for run in runs]) if not dry_run else None
                    extras = np.array([run[1] for run in runs]) if not dry_run else None
                else:
                    L = np.array(runs) if not dry_run else None

                idx = np.argsort(L) if not dry_run else None
                T = T[idx] if not dry_run else None
                L = L[idx] if not dry_run else None
                if early_stopping:
                    extras = extras[idx] if not dry_run else None

                if early_stopping:
                    for config, loss, extra in (zip(T, L, extras) if not dry_run else []):
                        results_total.append(dict(loss=loss, epochs=r_i, config=config,
                                                  extra=extra))
                    stopped = np.array([e['stopped_early'] for e in extras])
                    T = T[np.invert(stopped)]
                else:
                    for config, loss in (zip(T, L) if not dry_run else []):
                        results_total.append(dict(loss=loss, epochs=r_i, config=config))

                T = T[:int(math.ceil(n_i / self.eta))] if not dry_run else None
                log("keeping: {0}".format(math.ceil(n_i / self.eta)))

        log("runs total: {0}".format(runs_total))
        return results_total


if __name__ == "__main__":
    hyperband = Hyperband(None, None, max_epochs=300, reduction_factor=3, min_r=5)
    # hyperband = Hyperband(None, None, max_epochs=1500, reduction_factor=5, min_r=5)
    hyperband.run(verbose=True, dry_run=True, early_stopping=False)
    # hyperband = Hyperband(lambda: dict(test=np.random.randint(0, 2)),
    #                       lambda x, r: (np.random.rand(1)[0],
    #                                     dict(stopped_early=bool(np.random.randint(0, 2)))),
    #                       max_epochs=15, reduction_factor=5, min_r=5)
    # hyperband.run(verbose=True, dry_run=False, early_stopping=True)
