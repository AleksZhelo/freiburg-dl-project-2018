from __future__ import print_function

# import os
#
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
from operator import itemgetter
from util.loader import load_data_as_numpy
from optimization.run_model import run_model

import tensorflow as tf
import gc

from optimization.mlp import MLP

configs, learning_curves = load_data_as_numpy()

# GA Parameters
NUM_POP = 20  # Population size (keep it even)
X_RATE = 0.5  # Selection rate
MUT_RATE = 0.2  # Mutation rate
NUM_ELITE = 4  # Number of elite chromosomes to keep (even number)
MAX_IT = 500  # Max number of iterations
NUM_KEEP = 14  # np.ceil(X_RATE*NUM_POP)        /// even number >= 2 >= NUM_ELITE

# # Threading params
# THREADS = 8
# THREAD_DELAY = 0.001

# Other params
results = {}

# config = tf.ConfigProto(
#         device_count={'GPU': 0}
#     )

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True

# Functions


# Evaluate a chromosomes fitness
def evaluate(chromosome):
    code = str(chromosome)
    if code in results:
        return chromosome, results[code]
    else:
        return eval_model(chromosome)


# Convert chromosome to params (list to dict)
def build_params(c):
    params = {}
    for g in c:
        params[g[0]] = g[1][0]
    return params


# TODO: Consider giving points for training time?
def eval_model(c):
    with tf.Session() as session:
        # if decay_lr:
        #     model.append_decay_params(params, rs, configs.shape[0] / batch_size)
        params = build_params(c)  # to dict
        model = MLP
        normalize = params["normalize"]
        eval_every = params["eval_n"]
        batch_size = params["batch_size"]
        train_epochs = params["epochs"]
        cv_loss = run_model(session, configs, learning_curves, None,
                            model, normalize, train_epochs, batch_size, eval_every, params, verbose=False)
        code = str(c)
        results[code] = cv_loss
    tf.reset_default_graph()
    gc.collect()
    return c, cv_loss

# Population selections/sorting


def sort_pop(pop):
    """
    Sorts a population
    :param pop: population
    :return: sorted list of (chrom, value) pair
    """
    npop = list(map(evaluate, pop))
    return sorted(npop, key=itemgetter(1))  # reverse=True (if maximizing)


def to_pop(dpop):
    """
    Converts list of (chrom, value) pair to a population
    :param dpop: (chrom, value) pair
    :return: population
    """
    return list(map(lambda x: x[0], dpop))


# Returns a selection (keep-elites) from a population
def selection(spop):
    return spop[NUM_ELITE:NUM_KEEP]


# Returns a selection (keep) from a population
def select_parents(spop):
    return spop[0:NUM_KEEP]


# Returns a selection (elites) from a population
def select_elites(spop):
    return spop[0:NUM_ELITE]

# Evolution functions


# Evolves a population n times
def evolve_population(n, pop):
    npop = pop
    for i in range(n):
        npop = evolve_once(npop)
        print(results)
        best = results[str(npop[0])]
        print("\rEvolved {}/{} ({})".format(i + 1, n, best), end="")
        # TODO: Logging goes here
    return to_pop(sort_pop(npop))


# Evolves a population one time
def evolve_once(pop):
    parents = to_pop(select_parents(sort_pop(pop)))
    offspring = mate_pairwise(parents)
    parents.extend(offspring)
    return mutate_population(parents)


# Chromosome crossover (could also consider doing multiple points)
def crossover(cp, ma, pa):
    nc = ma[:cp]
    nc.extend(pa[cp:])
    return nc


# Mate a population pairwise (1+2, 3+4 etc)
def mate_pairwise(pop):
    if not pop:  # Empty
        return []
    npop = []
    ma = pop[0]
    pa = pop[1]
    nc = crossover(np.random.randint(len(ma)), ma, pa)
    npop.append(nc)
    npop.extend(mate_pairwise(pop[2:]))
    return npop


# Mutate an entire population based on mutation rate
def mutate_population(pop):
    indices = list(np.random.randint(NUM_ELITE, len(pop), int(np.ceil(MUT_RATE*NUM_POP))))
    for i in indices:
        pop[i] = mutate_chromosome(pop[i])
    return pop


# Mutate a random gene in a single chromosome
def mutate_chromosome(chrom):
    indice = np.random.randint(len(chrom))
    n, gene = chrom[indice]
    chrom[indice] = n, mutate_gene(gene)
    return chrom


# Mutate a gene (currently creates a random new one, could be altered to sometimes be based on previous gene)
def mutate_gene(gene):
    return gene[1](gene[2]), gene[1], gene[2]


def random_layer(args):
    return 2 ** np.random.randint(args[0], args[1]), np.random.random()  # neurons, dropout_rate


def random_exp(args):
    return np.exp(np.random.randint(args[0], args[1]))


def random_multiple(args):
    return args[0] ** np.random.randint(args[1], args[2])


def random_bool(args):
    return np.random.choice([True, False])


def random_int(args):
    return np.random.randint(args[0], args[1])


def initialize_population():
    pop = []
    for i in range(NUM_POP):
        pop.append(initialize_chromosome())
    return pop


def initialize_chromosome():
    c = [
        ("l1", mutate_gene((0, random_layer, [1, 12]))),  # First layer
        ("l2", mutate_gene((0, random_layer, [0, 12]))),  # Second layer
        ("l3", mutate_gene((0, random_layer, [0, 12]))),  # Third layer
        ("l4", mutate_gene((0, random_layer, [0, 12]))),  # Fourth layer
        ("l5", mutate_gene((0, random_layer, [0, 12]))),  # Fifth layer
        ("learning_rate", mutate_gene((0, random_multiple, [10, -8, 0]))),  # Learning rate, random exp?
        ("batch_size", mutate_gene((0, random_multiple, [2, 1, 10]))),  # Batch size (2**9 = 512)
        ("dropout", mutate_gene((0, random_bool, []))),  # Use dropout?
        ("normalize", mutate_gene((0, random_bool, []))),  # Normalize?
        ("eval_n", mutate_gene((0, random_int, [4, 5]))),  # Evaluate every n (static 4 for now)
        ("epochs", mutate_gene((0, random_int, [10, 400]))),  # Epochs
        ]
    return c


if __name__ == "__main__":
    pop = initialize_population()
    pop = evolve_population(MAX_IT, pop)
    print("=========")
    print(pop[0])  # Best result


