import os

import numpy as np
import pandas as pd

from tensorflow import math
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

from src.dataset_creation.label_encoding import label_to_int, int_to_string_label


def plot_confusion_matrix(cm, label_type, normalize, name, experiments_dir):
    """
    Function to plot the confusion matrix

    :param cm: numpy array with prediction per column and true label per row
    :param label_type: string, 'isup' (needed for axis annotation)
    :param normalize: bool, use relative or absolute values
    :param name: filename
    :param experiments_dir: directory where to save the plot
    :return:
    """
    size = cm.shape[0]
    if normalize:
        precision = float
        for i in range(size):
            if sum(cm[i]) > 0:
                cm[i] = cm[i] / sum(cm[i])
            else:
                cm[i] = -1
    else:
        precision = int
    # Limits for the extent
    xy_start = 0.0
    xy_end = size
    extent = [xy_start, xy_end, xy_start, xy_end]
    # The normal figure
    f = plt.figure(figsize=(16, 12))
    ax = f.add_subplot(111)
    im = ax.imshow(cm, extent=extent,  cmap='Blues')
    # Add the text
    jump_xy = (xy_end - xy_start) / (2.0 * size)
    xy_positions = np.linspace(start=xy_start, stop=xy_end, num=size, endpoint=False)
    # Change color of text to white above threshold
    max_value = np.sum(cm, axis=1).max()
    thresh = max_value / 2.
    for y_index, y in enumerate(reversed(xy_positions)):
        for x_index, x in enumerate(xy_positions):
            text_x = x + jump_xy
            text_y = y + jump_xy
            if cm[y_index, x_index] == -1:
                label = '-'
                ax.text(text_x, text_y, label, color="black", ha='center', va='center', fontdict={'size': 30})
            else:
                label = precision(round(cm[y_index, x_index]*100)/100)
                ax.text(text_x, text_y, label, color="white" if label > thresh else "black", ha='center', va='center', fontdict={'size': 30})
    if precision == float:
        color_range_ticks = np.linspace(0, max_value, 6, endpoint=True)
    else:
        color_range_ticks = np.linspace(0, max_value, 5, endpoint=True)

    im.set_clim(0, max_value)
    cbar = f.colorbar(im, ticks=color_range_ticks)
    cbar.ax.tick_params(labelsize=30)
    plt.xlabel('prediction', fontdict={'size': 30})
    plt.ylabel('ground truth', fontdict={'size': 30})

    ticklabels = [int_to_string_label(label_int, label_type) for label_int in range(len(cm))]
    plt.xticks(np.arange(xy_start + 0.5, xy_end), ticklabels, fontsize=30, rotation=20, horizontalalignment='right')
    plt.yticks(np.arange(xy_start + 0.5, xy_end), ticklabels[::-1], fontsize=30)
    
    plt.tight_layout()
    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', experiments_dir))
    f.savefig(os.path.join(script_path, name+'.png'))
    
    return


def create_classification_result_dataframe(model, dataset, class_distribution, label_type, n_output_nodes):
    """
    Function to create a dataframe with the final results for each image

    :param model: tensorflow/keras model
    :param dataset: tf.data.Dataset with entries like dictionary, needs to include
                    'images': tf.Tensor [batch_size(1), w, h, 3]
                    'labels': tf.Tensor [batch_size(1), n_output_nodes]
    :param class_distribution: list with length n_classes, and elements how many examples in dataset per class
    :param label_type: string 'isup' or 'bin'
    :param n_output_nodes: int, how many output nodes the model has
    :return: results as pandas Dataframe, with columns
             img_path: string
             groundtruth_class: int
             predicted_class: int
             pred_class_x: one column per output node, returned prediction (float) per output node
             any additional labels that were defined during dataset creation
    """
    assert dataset.element_spec['labels'].shape[-1] == n_output_nodes == model.layers[-1].output_shape[-1]

    # go through all examples and find predicted value
    n_examples = int(sum(class_distribution))

    targets = np.zeros((n_examples, n_output_nodes))
    predictions = np.zeros((n_examples, n_output_nodes))
    img_names = [''] * n_examples
    for idx, d in enumerate(dataset.take(n_examples)):
        if idx == 0:
            additional_labels = {k: list() for k in d if k not in ['images', 'labels']}
        img = d['images']
        label = d['labels']

        # run model to get prediction
        pred = model(img, training=False).numpy()
        # only use first element, because one element with batch size 1 is used
        pred = pred[0]

        predictions[idx, :] = pred
        targets[idx, :] = label[0]
        img_names[idx] = d['image_paths'].numpy()[0].decode('utf-8')
        [additional_labels[k].append(np.array(d[k]).squeeze()) for k in d if k not in ['images', 'labels']]

    results = pd.DataFrame()
    results['img_path'] = img_names
    results['groundtruth_class'] = label_to_int(targets, label_type)
    results['predicted_class'] = label_to_int(predictions, label_type)
    for pred_class in range(predictions.shape[1]):
        results['pred_class_'+str(pred_class)] = predictions[:, pred_class]
    for k in additional_labels:
        results[k] = list(additional_labels[k])

    return results


def evaluate_classification_model(model, dataset, class_distribution, label_type, metrics, experiments_dir):
    """
    Main evaluation function

    :param model: tensorflow/keras model
    :param dataset: tf.data.Dataset with entries like dictionary, needs to include
                    'images': tf.Tensor [batch_size(1), w, h, 3]
                    'labels': tf.Tensor [batch_size(1), n_output_nodes]
    :param class_distribution: list with length n_classes, and elements how many examples in dataset per class
    :param label_type: string 'isup' or 'bin'
    :param metrics: list of strings with metrics ['acc','f1','kappa']
    :param experiments_dir: path to where to store results, e.g. /path/to/experiments
    :return: resulting metrics as dictionary, depending on which were specified can include
             'accuracy': float between 0 and 1
             'cohens_kappa: float between -1 and 1
             'f1_score': float between 0 and 1
    """
    result_metrics = dict()
    n_classes = int(len(class_distribution))
    if label_type == 'isup':
        n_output_nodes = n_classes - 1
    else:
        n_output_nodes = n_classes

    results = create_classification_result_dataframe(model, dataset, class_distribution, label_type, n_output_nodes)
    results.to_csv(os.path.join(experiments_dir, 'results.csv'))

    if np.any(['acc' in m for m in metrics]):
        acc = accuracy_score(results['groundtruth_class'], results['predicted_class'])
        print("accuracy: ", acc)
        result_metrics['accuracy'] = float(acc)

    if np.any(["kappa" in m for m in metrics]):
        kappa = cohen_kappa_score(results['groundtruth_class'], results['predicted_class'], weights='quadratic')
        print("kappa: ", kappa)
        result_metrics['cohens_kappa'] = float(kappa)

    if np.any(['f1' in m for m in metrics]):
        f1 = f1_score(results['groundtruth_class'], results['predicted_class'], average='macro')
        print("F1 score: ", f1)
        result_metrics['f1_score'] = float(f1)

    # CONFUSION MATRIX
    mat = math.confusion_matrix(results['groundtruth_class'], results['predicted_class'], num_classes=n_classes)
    plot_confusion_matrix(np.array(mat, dtype='float32'), label_type, True, 'confusion_matrix_relative', experiments_dir)
    plot_confusion_matrix(np.array(mat), label_type, False, 'confusion_matrix', experiments_dir)
    
    result_metrics['cm'] = mat.numpy().astype(float).tolist()

    return result_metrics
