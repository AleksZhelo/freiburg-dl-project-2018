import matplotlib.pyplot as plt

colors = ["black", "blue", "red", "green", "purple"]

def scatter(y, y_hat, title = None, file_name = None):
    """
    y : list of true values
    y_hat : list of predicted values
    title : the plot's title
    file_name : path to file where the plot shall be saved
    """
    plt.figure(figsize=(6, 6))
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
                  title=None,
                  file_name=None):
    """
    true_curve : list with true validation errors per epoch
    extrapolation_list : list of tuples (label, extrapolation curve, number of true steps)
        an extrapolation curve of length k is plotted for the last k epochs
    n_steps : last epoch of the extrapolation curves
    file_name : path to file where the plot shall be saved
    """
    plt.plot(true_curve, color=colors[0])
    plt.xlim(0, n_steps - 1)
    for i, extrapolation in enumerate(extrapolation_list):
        extra_curve = list(extrapolation[1])
        extra_len = len(extra_curve)
        # add last point of true curve to predicted curve:
        if extra_len <= len(true_curve):
            extra_curve = [true_curve[n_steps - extra_len - 1]] + extra_curve
            extra_len += 1
        plt.plot(range(n_steps - extra_len, n_steps), extra_curve, color=colors[(i + 1) % len(colors)])
    
    # vertical lines
    y_limits = plt.gca().get_ylim()
    for i, extrapolation in enumerate(extrapolation_list):
        plt.vlines([extrapolation[2]], 0, 1, colors=[colors[i + 1]], linestyles="dashed")
    plt.ylim(y_limits)
    
    # axis labels
    plt.xlabel("epoch")
    plt.ylabel("validation error")
    
    # legend
    if true_curve[-1] > 0.8:
        legend_pos = "lower right"
    else:
        legend_pos = "upper right"
    plt.legend(["true"] + [extra[0] for extra in extrapolation_list],
               loc = legend_pos)
    
    # title
    if title != None:
        plt.title(title)
    
    # save plot
    if file_name != None:
        plt.savefig(file_name)
        plt.close()
    else:
        plt.show()

def boxplot(errors_list,
            squared=False,
            logarithmic=True,
            file_name=None):
    """
    errors_list : a list of tuples (label, list of errors)
    squared : True for squared errors, False for absolute errors
    logarithmic : whether y-axis should be logarithmic
    file_name : path to file where the plot shall be saved
    """
    for i in range(len(errors_list)):
        if squared:
            errors_list[i] = (errors_list[i][0], [val ** 2 for val in errors_list[i][1]])
        else:
            errors_list[i] = (errors_list[i][0], [abs(val) for val in errors_list[i][1]])
    plt.figure(figsize=(10, 6))
    ax = plt.subplot()
    ax.boxplot([errors[1] for errors in errors_list],
                     labels=[errors[0] for errors in errors_list])
    if logarithmic:
        ax.set_yscale("log")
    plt.ylabel("squared error" if squared else "absolute error")
    if file_name != None:
        plt.savefig(file_name)
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    import numpy as np
    n = 100
    y = np.random.uniform(size=n)
    y_hat = np.random.uniform(size=n)
    y_hat2 = np.random.uniform(size=n)
    
    plt.figure(0)
    scatter(y, y_hat, "test title")
    
    plt.figure(1)
    extrapolation(y[:50],
                  [("90 steps", y_hat2[-90:], 10),
                   ("50 steps", y_hat[-50:], 50)],
                  n_steps=100)
    
    plt.figure(2)
    boxplot([("test", [y[i] - y_hat[i] for i in range(n)])])
