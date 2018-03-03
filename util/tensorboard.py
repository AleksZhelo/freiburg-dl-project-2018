import tensorflow as tf

def tensorboard_plot(run_name, plot_name, values):
    summary_writer = tf.summary.FileWriter("logs/" + run_name)
    for i in range(len(values)):
        summary = tf.Summary()
        summary.value.add(tag=plot_name, simple_value=values[i])
        summary_writer.add_summary(summary, i)
    summary_writer.flush()

def tensorboard_log_values(run_name, epoch, plot_values):
    summary_writer = tf.summary.FileWriter("logs/" + run_name)
    summary = tf.Summary()
    for plot_name in plot_values:
        summary.value.add(tag=plot_name, simple_value=plot_values[plot_name])
    summary_writer.add_summary(summary, epoch)
    summary_writer.flush()