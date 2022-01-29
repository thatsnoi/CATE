import logging
import time
from datetime import datetime
from textwrap import wrap
import re
import itertools
import seaborn as sns
import io

from matplotlib import pyplot as plt

TRAINING_INFO_LEVEL_NUM = 15


def now_as_str_f():
    return "{:%Y_%m_%d---%H_%M_%f}".format(datetime.now())


def plot_confusion_matrix(correct_labels, predict_labels, labels, title='Confusion matrix',
                          tensor_name='MyFigure/image', normalize=False):
    '''
    Parameters:
        correct_labels                  : These are your true classification categories.
        predict_labels                  : These are you predicted classification categories
        labels                          : This is a lit of labels which will be used to display the axix labels
        title='Confusion matrix'        : Title for your matrix
        tensor_name = 'MyFigure/image'  : Name for the output summay tensor

    Returns:
        summary: TensorFlow summary

    Other itema to note:
        - Depending on the number of category and the data , you may have to modify the figzie, font sizes etc.
        - Currently, some of the ticks dont line up due to rotations.
    '''
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(correct_labels, predict_labels, labels=range(0, 28))
    np.seterr(divide='ignore', invalid='ignore')
    if normalize:
        cm = cm.astype('float') * 10 / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm, copy=True)
        cm = cm.astype('int')
    np.seterr(divide='warn', invalid='warn')

    np.set_printoptions(precision=2)
    ###fig, ax = matplotlib.figure.Figure()

    fig = plt.figure(figsize=(7, 7), dpi=320, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(cm, cmap='Oranges')

    classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in labels]
    classes = ['\n'.join(wrap(l, 40)) for l in classes]

    tick_marks = np.arange(len(classes))

    ax.set_xlabel('Predicted', fontsize=7)
    ax.set_xticks(tick_marks)
    c = ax.set_xticklabels(classes, fontsize=4, rotation=-90, ha='center')
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    ax.set_ylabel('True Label', fontsize=7)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=4, va='center')
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], 'd') if cm[i, j] != 0 else '.', horizontalalignment="center", fontsize=6,
                verticalalignment='center', color="black")
    fig.set_tight_layout(True)
    fig.canvas.draw()

    # Now we can save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return data


def now_as_str_f():
    return "{:%Y_%m_%d---%H_%M_%f}".format(datetime.now())


def get_logger(log_path=None, log_level=logging.DEBUG):
    logging.getLogger().setLevel(logging.WARNING)
    logger = logging.getLogger('uncertainty_estimation_in_dl')

    if log_path and not logger.handlers:
        # set the level
        logger.setLevel(log_level)

        # Logging to a file
        f = '[%(asctime)s][%(levelname)s][%(message)s]'
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter(fmt=f, datefmt='%d/%m-%H:%M:%S'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter(fmt=f, datefmt='%d/%m-%H:%M:%S'))
        logger.addHandler(stream_handler)

        # Add a new level between debug and info for printing logs while training
        logging.addLevelName(TRAINING_INFO_LEVEL_NUM, 'TINFO')
        setattr(logger, 'tinfo', lambda *args: logger.log(TRAINING_INFO_LEVEL_NUM, *args))

    return logger


def show_values(pc, fmt="%.2f", **kw):
    '''
    Heatmap with text in each cell with matplotlib's pyplot
    Source: https://stackoverflow.com/a/25074150/395857
    By HYRY
    '''
    from itertools import izip
    pc.update_scalarmappable()
    ax = pc.get_axes()
    # ax = pc.axes# FOR LATEST MATPLOTLIB
    # Use zip BELOW IN PYTHON 3
    for p, color, value in izip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
        x, y = p.vertices[:-2, :].mean(0)
        if np.all(color[:3] > 0.5):
            color = (0.0, 0.0, 0.0)
        else:
            color = (1.0, 1.0, 1.0)
        ax.text(x, y, fmt % value, ha="center", va="center", color=color, **kw)

def plot_classification_report(correct_labels, predict_labels, labels):
    from sklearn.metrics import classification_report
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd

    clf_report = classification_report(correct_labels,
                                       predict_labels,
                                       labels=range(0, 28),
                                       target_names=labels,
                                       output_dict=True,
                                       zero_division=0)
    print(clf_report)
    fig = None
    cr_ax = sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True, fmt='.2f', xticklabels=True, yticklabels=True)
    fig = cr_ax.get_figure()
    fig.set_size_inches(10.5, 10.5)
    fig.set_tight_layout(True)
    fig.canvas.draw()

    # Now we can save it to a numpy array.
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return image_from_plot

def plot_axes(ax, fig=None, geometry=(1,1,1)):
    if fig is None:
        fig = plt.figure()
    if ax.get_geometry() != geometry:
        ax.change_geometry(*geometry)
    ax = fig.axes.append(ax)
    plt.close(fig)
    return fig