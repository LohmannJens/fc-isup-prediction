import numpy as np
import tensorflow as tf


def transform_labels(labels):
    """
    This script takes a list of labels as input and transforms the list to a list of for example one-hot vectors.
    Which label (string) maps to which number in the resulting vector, needs to be defined in a single file, which has
    the name as in data_generation_config['dataset_label']

    This is not specific for gleason, but can be used for any other label, too

    For survival prediction, the labels can be automatically transformed without the need for an extra file,
    when the length and amount of intervals are known

    :param labels: a list of labels as strings, in whichever form. E.g. ['20.3','80.4', ...]
    :return: a list of labels in the desired format. E.g. [[1 1 0 0 0 0], [1 1 1 0 0 0], [0 0 0 0 0 0]]
             and array of labels as integer # the class_distribution
    """
    label_type = "isup"
    n_classes = 6
    n_output_nodes = n_classes - 1

    try:
        labels = [float(l) for l in labels]
    except:
        if label_type == 'isup':
            labels = [convert_gleason_to_isup(l) for l in labels]
    labels = np.array(labels)[..., np.newaxis]

    if label_type == 'isup':
        labels_encoded = labels > np.arange(n_output_nodes)
        interval_limits = None
        label_array = np.sum(labels_encoded, axis=1)
    else:
        raise NotImplementedError('only label_types "isup", "bin" or "survival" are supported')
    return np.array(labels_encoded, dtype='float32'), label_array, interval_limits


def convert_gleason_to_isup(gleason):
    gleason_isup_dict = {'0': 0,
                         '3+3': 1,
                         '3+4': 2,
                         '3+5': 4,
                         '4+3': 3,
                         '4+4': 4,
                         '4+5': 5,
                         '5+3': 4,
                         '5+4': 5,
                         '5+5': 5}
    return gleason_isup_dict[gleason]


@tf.function
def tf_label_to_int(label, encoding):
    return tf.numpy_function(label_to_int, [label, encoding], tf.float32)


def label_to_int(label_batch, label_type):
    """
    Transform the label to a single integer, for isup and survival, this is the sum of the vector,
                                             for binary prediction, max of both values

    :param label_batch: a batch of encoded labels, e.g. [[1 0 0 0] [0 0 1 0]]
    :param label_type: either
                        bin: [1,0] -> 0 [0,1] -> 1
                        isup: [0,0,0,0,0] -> 0 [1,1,1,0,0] -> 3
                        survival: [1,0,0,0,0,0] -> 1 (same as isup)
                        TODO: include option for Cox ?!
    :return: label integers, e.g. [0 2]
    """
    if isinstance(label_type, bytes):
        label_type = label_type.decode('utf-8')
    if label_type == 'isup':
        if isinstance(label_batch, np.ndarray):
            return np.round(np.sum(label_batch, axis=1))
        else:
            return tf.round(tf.reduce_sum(label_batch, axis=1))
    else:
        raise NotImplementedError('Right now, only label_encodings "bin, isup, survival" available')


def int_to_string_label(int_label, label_type):
    """
    Used for legend of confusion matrix
    
    :param int_label: int of class
    :param label_type: string, isup or bin
    :return: string (int_label converted to string)
    """
    if label_type == 'isup':
        label_list = ['0', '3+3 / 1', '3+4 / 2', '4+3 / 3', '3+5/4+4/5+3 / 4', '34+5/5+4/5+5 / 5']
        return label_list[int_label]
    else:
        return str(int_label)
