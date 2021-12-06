"""
Miscellaneous utilities to read and work with the json files and such.
Stuff multiple functions use.
"""
import sys
import os
import errno
import json
import logging
import numpy as np
# Use mpl on remote.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def print_sorted_dict(d, out_file):
    for k in sorted(d, key=d.get, reverse=True):
        try:
            out_file.write("{}, {}\n".format(k, d[k]))
        except UnicodeError:
            pass


def create_dir(dir_name):
    """
    Create the directory whose name is passed.
    :param dir_name: String saying the name of directory to create.
    :return: None.
    """
    # Create output directory if it doesnt exist.
    try:
        os.makedirs(dir_name)
        print('Created: {}.'.format(dir_name))
    except OSError as ose:
        # For the case of *file* by name of out_dir existing
        if (not os.path.isdir(dir_name)) and (ose.errno == errno.EEXIST):
            sys.stderr.write('IO ERROR: Could not create output directory\n')
            sys.exit(1)
        # If its something else you don't know; report it and exit.
        if ose.errno != errno.EEXIST:
            sys.stderr.write('OS ERROR: {:d}: {:s}: {:s}\n'.format(ose.errno,
                                                                   ose.strerror,
                                                                   dir_name))
            sys.exit(1)


def read_json(json_file):
    """
    Read per line JSON and yield.
    :param json_file: File-like with a next() method.
    :return: yield one json object.
    """
    for json_line in json_file:
        # Try to manually skip bad chars.
        # https://stackoverflow.com/a/9295597/3262406
        try:
            f_dict = json.loads(json_line.replace('\r\n', '\\r\\n'),
                                encoding='utf-8')
            yield f_dict
        # Skip case which crazy escape characters.
        except ValueError:
            raise


def plot_train_hist(y_vals, checked_iters, fig_path, ylabel, suffix=None):
    """
    Plot y_vals against the number of iterations.
    :param y_vals: list; values along the y-axis.
    :param checked_iters: list; len(y_vals)==len(checked_iters); the iterations
        the values in y_vals correspond to.
    :param fig_path: string; the directory to write the plots to.
    :param ylabel: string; the label for the y-axis.
    :param suffix: string; string to add to the figure filename.
    :return: None.
    """
    # If there is nothing to plot just return.
    if len(checked_iters) <= 3:
        return
    x_vals = np.array(checked_iters)
    y_vals = np.vstack(y_vals)
    plt.plot(x_vals, y_vals, '-', linewidth=2)
    plt.xlabel('Training iteration')
    plt.ylabel(ylabel)
    plt.title('Evaluated every: {:d} iterations'.format(
        checked_iters[1]-checked_iters[0]))
    plt.tight_layout()
    ylabel = '_'.join(ylabel.lower().split())
    if suffix:
        fig_file = os.path.join(fig_path, '{:s}_history-{:s}.eps'.format(ylabel, suffix))
    else:
        fig_file = os.path.join(fig_path, '{:s}_history.eps'.format(ylabel))
    plt.savefig(fig_file)
    if suffix:
        plt.savefig(os.path.join(fig_path, '{:s}_history-{:s}.png'.format(ylabel, suffix)))
    else:
        plt.savefig(os.path.join(fig_path, '{:s}_history.png'.format(ylabel)))
    plt.clf()
    logging.info('Wrote: {:s}'.format(fig_file))

