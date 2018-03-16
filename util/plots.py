import matplotlib.pyplot as plt

def scatter(y, y_hat, title = None, file_name = None):
    """
    y : list of true values
    y_hat : list of predicted values
    title : the plot's title
    file_name : path to file where the plot shall be saved
    """
    plt.scatter(y, y_hat)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.plot([0, 1], [0, 1], color="grey")
    plt.xlabel("true validation error")
    plt.ylabel("predicted validation error")
    if title != None:
        plt.title(title)
    if file_name != None:
        plt.savefig(file_name)
        plt.close()
    else:
        plt.show()

def extrapolation(true_curve,
                  extrapolation_list,
                  n_steps=40,
                  file_name=None):
    """
    true_curve : list with true validation errors per epoch
    extrapolation_list : list of tuples (label, extrapolation curve)
        an extrapolation curve of length k is plotted for the last k epochs
    n_steps : last epoch of the extrapolation curves
    file_name : path to file where the plot shall be saved
    """
    plt.plot(true_curve)
    plt.xlim(0, n_steps - 1)
    for extrapolation in extrapolation_list:
        extra_curve = list(extrapolation[1])
        extra_len = len(extra_curve)
        # add last point of true curve to predicted curve:
        extra_curve = [true_curve[n_steps - extra_len - 1]] + extra_curve
        extra_len += 1
        plt.plot(range(n_steps - extra_len, n_steps), extra_curve)
    plt.xlabel("epoch")
    plt.ylabel("validation error")
    if true_curve[-1] > 0.8:
        legend_pos = "lower right"
    else:
        legend_pos = "upper right"
    plt.legend(["true"] + [extra[0] for extra in extrapolation_list],
               loc = legend_pos)
    if file_name != None:
        plt.savefig(file_name)
        plt.close()
    else:
        plt.show()