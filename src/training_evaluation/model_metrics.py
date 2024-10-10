import numpy as np
import tensorflow as tf

from sklearn.metrics import f1_score, cohen_kappa_score


def tf_f1_score_wrap(class_weights):
    """
    F1 score for classification

    :param class_weights: None or array with weight per class
    :return: F1-score
    """

    def tf_f1_score(y_true, y_pred):
        # wrapper for numpy function
        return tf.numpy_function(f1_score_core, [y_true, y_pred], tf.double)

    def f1_score_core(y_true, y_pred):
        """
        :param y_true: np array that is encoded as ordinal regression, e.g. [[1,1,1,0,0]] for class [3]
        :param y_pred: np array that is encoded as ordinal regression, e.g. [[1,1,1,0,0]] for class [3]

        :return: single float
        """
        y_true = np.round(np.sum(y_true, axis=1))
        y_pred = np.round(np.sum(y_pred, axis=1))
        f1 = f1_score(y_true, y_pred, average='macro')
        return f1
    return tf_f1_score


def cohens_kappa_wrap(class_weights):
    """
    Cohens kappa

    :param class_weights: None or array with weight per class
    :return: cohens kappa
    """
    def cohens_kappa(y_true, y_pred):
        # wrapper for numpy function
        return tf.numpy_function(kappa_core, [y_true, y_pred], tf.double)

    def kappa_core(y_true, y_pred):
        if len(y_true.shape) != 1:
            y_true = np.round(np.sum(y_true, axis=1))
            y_pred = np.round(np.sum(y_pred, axis=1))
        if class_weights is not None:
            sample_weight = [class_weights[y_true[i]] for i in range(y_true.shape[0])]
        else:
            sample_weight = None

        kappa = cohen_kappa_score(y_true, y_pred, weights='quadratic', sample_weight=sample_weight)
        return kappa

    return cohens_kappa


class TfCategoricalAccuracy(tf.keras.metrics.Metric):
    """
    Categorical accuracy needs to be defined new, because the resulting class in this case is not max(vector) but sum()
    """
    def __init__(self, name='tf_categorical_accuracy', **kwargs):
        super(TfCategoricalAccuracy, self).__init__(name=name, **kwargs)
        self.metr = 0

    def update_state(self, y_true, y_pred, class_weights=None):
        if (len(y_true.shape) != 1):
            y_true = np.round(np.sum(y_true, axis=1))
            y_pred = np.round(np.sum(y_pred, axis=1))
        if class_weights is not None:
            sample_weight = [class_weights[int(y_true[i])] for i in range(y_true.shape[0])]
        else:
            sample_weight = None
        acc = np.average(tf.equal(y_true, y_pred), weights=sample_weight)
        self.metr = acc

    def result(self):
        return self.metr


def tf_categorical_accuracy_wrap(class_weights):
    """
    Calculates categorical accuracy

    :param class_weights: None or array with weight per class
    :return: categorical accuracy
    """
    def tf_categorical_accuracy(y_true, y_pred):
        # wrapper for numpy function
        return tf.numpy_function(categorical_accuracy_core, [y_true, y_pred], tf.double)

    def categorical_accuracy_core(y_true, y_pred):
        if (len(y_true.shape) != 1):
            y_true = np.round(np.sum(y_true, axis=1))
            y_pred = np.round(np.sum(y_pred, axis=1))
        if class_weights is not None:
            sample_weight = [class_weights[int(y_true[i])] for i in range(y_true.shape[0])]
        else:
            sample_weight = None
        acc = np.average(tf.equal(y_true, y_pred), weights=sample_weight)
        return acc
    return tf_categorical_accuracy
