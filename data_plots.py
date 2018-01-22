import matplotlib.pyplot as plt
import numpy as np

from util.loader import load_data

configs, learning_curves = load_data(source_dir='./data')

plot_index = 0

y = [curve[-1] for curve in learning_curves]

for column_name in configs[0]:    
    x = [configs[i][column_name] for i in range(len(configs))]
    if column_name.startswith("log2"):
        x = [np.power(2, v) for v in x]
        column_name = column_name.replace("log2", "")
    plt.figure(plot_index)
    plt.plot(x, y, 'x')
    plt.xlabel(column_name)
    plt.ylabel("validation error")
    plt.show()
    plt.savefig("figures/" + column_name + ".png")
    plot_index += 1

for i in [0, 4, 9, 19, 29]:
    x = [curve[i] for curve in learning_curves]
    plt.figure(plot_index)
    plt.plot(x, y, 'x')
    plt.xlabel("learning_curve[" + str(i) + "]")
    plt.ylabel("validation error")
    plt.show()
    plt.savefig("figures/learning_curve" + str(i) + ".png")
    plot_index += 1