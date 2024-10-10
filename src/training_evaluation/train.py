import os

import numpy as np
import tensorflow as tf

from tensorflow.keras.callbacks import Callback

from src.dataset_creation.label_encoding import label_to_int


def _value_name_in_logs(logs, value):
    """
        For each client a number is added to the end of the value names.
        This is for example for metrics and monitor values the case.
        This function checks which number was added for the corresponding client.
        The function returns the new name of the value.
        This works for up to 20 clients.
        :param logs: logs of the run
        :param value: 
    """
    for i in range(21):
        if i == 0:
            new_val = value
        else:
            new_val =  f"{value}_{i}"
        if logs.get(new_val) or logs.get(new_val) == 0.0:
            return new_val
    print(f"value {value} does not exist")
    return

class LogPerformance(Callback):
    """
    Self defined Callback function, which at the moment calls an internal function at the end of each training epoch
    to save each model
    """
    def __init__(self, train_params):
        """
        Initialization

        :param train_params: dictionary with 'monitor_val': string (which metric/loss to use to decide for best model)
        :return:
        """
        self.monitor = train_params['monitor_val']

        super().__init__()
    def on_epoch_end(self, data, logs):
        """
        At end of each epoch, it should be evaluated whether the model is better now (based on monitor_val) and should
        be saved or not
        
        :param data: tupel with:
                        current epoch, needed to save model
                        path where to save the model
        :param logs: current metrics and losses
        :return:
        """
        epoch = data[0]
        path = data[1]
        model_save_name = self.monitor.split('_')[-1][:5]
        self.model.save(os.path.join(path, 'temp_weights{:03d}_{:s}{:.3f}.hdf5'.format(epoch, model_save_name, logs.get(self.monitor))))


@tf.function
def fiveepochlower(epoch, lr):
    """
    Halve learning rate every five epochs

    :param epoch: int, current epoch
    :param lr: float, current learning rate
    :return: float, updated learning rate
    """
    if (epoch % 5 == 0) and epoch != 0:
        lr = lr/2
    return lr


def tenepochlower(epoch, lr):
    """
    Halve learning rate every ten epochs

    :param epoch: int, current epoch
    :param lr: float, current learning rate
    :return: float, updated learning rate
    """
    if (epoch % 10 == 0) and epoch != 0:
        lr = lr/2
    return lr


def list_callbacks(train_params):
    """
    This function returns a customized callback function (which logs the metrics) and can also add more standard
    keras callbacks

    :param train_params: _config["train"] or a dictionary with info about epochs, callbacks, ...
    :return: list of callbacks
    """
    callbacks = list()
    callbacks.append(LogPerformance(train_params))
    valid_schedulers = {'tenepochlower': tenepochlower, 'fiveepochlower': fiveepochlower}

    if train_params['callbacks'] is not None:
        for t in train_params['callbacks']:
            if t['name'] == 'LearningRateScheduler':
                callbacks.append(getattr(tf.keras.callbacks, t['name'])(schedule=valid_schedulers[t['params']['schedule']]))
            else:
                callbacks.append(getattr(tf.keras.callbacks, t['name'])(**t['params']))
    return callbacks


def training_step(model, datapoint, class_weights, label_type):
    """
    Run model on input data, compute the loss and update the weights

    :param model: tensorflow model
    :param datapoint: datapoint as (tf.data) dict with labels and images
    :param class_weights: None or list of how many examples of each class exist, in order to weight samples
    :param label_type: 'isup'
    :return: prediction tf.Tensor and loss tf.Tensor (scalar value though)
    """
    label = datapoint['labels']
    img = datapoint['images']
    if class_weights is not None:
        class_weights = tf.convert_to_tensor([class_weights[k] for k in class_weights])
        int_of_class_weights = tf.cast([label_to_int(label, label_type)[i] for i in range(label.shape[0])], 'int32')
        sample_weights = tf.cast(tf.gather(class_weights, int_of_class_weights), 'float32')
    else:
        sample_weights = None

    with tf.GradientTape() as tape:
        prediction = model(img, training=True)
        loss = model.loss(y_true=label, y_pred=prediction)
        if sample_weights is not None:
            loss = loss * sample_weights
        loss = tf.reduce_mean(loss)
    gradients = tape.gradient(loss, model.trainable_variables)
    clip_value = 1.0
    gradients, _ = tf.clip_by_global_norm(gradients, clip_value)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return prediction, loss


@tf.function
def valid_step(model, datapoint, class_weights, label_encoding):
    """
    Validation step: run model on batch and evaluate loss, no weight update

    :param model: tensorflow/keras model
    :param datapoint: dictionary with
                      'images' tf.Tensor [batchsize, h, w, 3] and
                     'labels' tf.Tensor [batch_size, n_classes (maybe+1 for censoring information)]
    :param class_weights: None or array with weight per class
    :param label_encoding: 'isup'

    :return: prediction tf.Tensor and loss tf.Tensor (scalar value though)
    """
    label = datapoint['labels']
    img = datapoint['images']
    if class_weights is not None:
        class_weights = tf.convert_to_tensor([class_weights[k] for k in class_weights])
        int_of_class_weights = tf.cast([label_to_int(label, label_encoding)[i] for i in range(label.shape[0])], 'int32')
        sample_weights = tf.cast(tf.gather(class_weights, int_of_class_weights), 'float32')
    else:
        sample_weights = None
    prediction = model(img, training=False)

    loss = model.loss(y_true=label, y_pred=prediction)
    if sample_weights is not None:
        loss = loss * sample_weights
    loss = tf.reduce_mean(loss)

    return prediction, loss


def setup(model):
    """
    For training and validation loop, the loss and model (re)set each epoch

    :param model: tensorflow model, for which the metrics should be reset
    :return: loss and model with initialized metric states
    """
    loss = 0
    for metric in model.metrics:
        metric.reset_states()
    for metric in model.compiled_metrics._metrics:
        metric.reset_states()
    return loss, model


def update_standard_metrics(model, data_batch, prediction_batch):
    """
    Most metrics can be updated here, only some are left out (the ones that require more than a few data points)
    
    :param model: tensorflow / keras model
    :param data_batch: dictionary with 'images': tf.Tensor and 'labels': tf.Tensor
    :param prediction_batch: tf.Tensor (batch_size, n_classes)
    :return:
    """
    batch_size = prediction_batch.shape[0]
    for metric in model.metrics:
        label_batch = data_batch['labels']
        if metric.name in ['cohens_kappa'] and batch_size == 1:
            pass  # only calculate in the end
        elif metric.name in ['c_index_censor']:
            pass
        else:
            if 'censor' in metric.name:
                censored_batch = data_batch['censored']
                label_batch = tf.concat((np.array(label_batch, 'float32'), tf.expand_dims(np.array(censored_batch, 'float32'), 1)), 1)
            metric.update_state(label_batch, prediction_batch)

    if len(model.metrics) == 0:
        for metric in model.compiled_metrics._metrics:
            label_batch = data_batch['labels']
            metric.update_state(label_batch, prediction_batch)
