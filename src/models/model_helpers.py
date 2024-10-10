import os
import logging

import numpy as np
import tensorflow as tf

from src.training_evaluation.model_metrics import TfCategoricalAccuracy
import src.training_evaluation.model_losses as own_losses


def m_isup(model_config):
    """
    M_ISUP to predict one out of 5 ISUP classes or benign from image, output is encoded with ordinal regression,
        therefore, number of output nodes is one less than number of classes

    :param model_config: dictionary with
                         base_model: e.g. string that is either a model that can be loaded from tf.keras.applications
                                    (like "InceptionV3") or it is the path to a folder with a self-pretrained model
                         for loading the model from tf.keras (input_shape: [m,n,3],
                                                              weights:'imagenet' / None)
    :return: tensorflow model
    """
    # first, load the base model
    bmodel, x = load_base_model(model_config)
    n_classes = 6 #model_config['n_classes']

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    if (model_config['dense_layer_nodes'] is not None) and (model_config['dense_layer_nodes'] != False):
        for n_nodes in model_config['dense_layer_nodes']:
            x = tf.keras.layers.Dense(units=n_nodes)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dense(units=n_classes-1, activation='sigmoid')(x)
    model = tf.keras.models.Model(bmodel.input, x)
    return model


def load_base_model(model_config):
    """
    Either load a model predefined in tf.keras (pretrained or not) or load an own model

    :param model_config: config file with model parameters
    :return: base model and either cutoff layer or last layer 
    """
    try:
        bmodel = tf.keras.models.load_model(os.path.join("/mnt", "input", "InceptionV3.hdf5"))
    except:
        print(f"Add InceptionV3.hdf file to shared folder.")
    
    for layer in bmodel.layers:
        try:
            layer._name = layer._name + str('_base')
        except:
            print("Could not transfer weights for layer {}".format(layer.name))

    if 'cut_off_layer' in model_config:
        cut_off_after_layer = model_config['cut_off_layer']
    else:
        cut_off_after_layer = False
    if isinstance(cut_off_after_layer, int) and not isinstance(cut_off_after_layer, bool):
        x = bmodel.layers[cut_off_after_layer].output
    elif isinstance(cut_off_after_layer, str):
        x = bmodel.get_layer(cut_off_after_layer).output
    else:
        x = bmodel.layers[-1].output
    return bmodel, x


def compile_model(model, train_params, metric_class_weights=None, _log=logging):
    """
    Compiles the selected model

    :param model: tensorflow/keras model
    :param train_params: dictionary with 'optimizer.name'
    :param metric_class_weights: None or array with weight per class
    :param _log: log 
    :return: compile model
    """
    try:
        # parameters are a list, so not unintentionally overwritten by main config
        optimizer = getattr(tf.keras.optimizers, train_params["optimizer"]['name'])(
                **{k: v for d in train_params['optimizer']['params'] for k, v in d.items()})
    except AttributeError:
        raise NotImplementedError("only optimizers available at tf.keras.optimizers are implemented at the moment")

    try:
        loss = getattr(tf.keras.losses,  train_params["loss_fn"])
    except AttributeError:
        try:
            loss = getattr(own_losses, train_params['loss_fn']+'_wrap')()
        except AttributeError:
            raise NotImplementedError("only losses "
                                      "available at tf.keras.losses or "
                                      "cdor, deepconvsurv and ecarenet_loss "
                                      "are implemented at the moment")
    except TypeError:
        print('loss_information', *train_params['loss_fn'])
        loss = getattr(own_losses, train_params['loss_fn'][0]+'_wrap')(*train_params['loss_fn'][1:])

    # read all metrics from the list in config.yaml
    metrics = list()
    if train_params['compile_metrics'] is not None:
        for metric in train_params["compile_metrics"]:
            try:
                metrics.append(getattr(tf.keras.metrics, metric)())
            except AttributeError:
                try:
                    if metric == "TfCategoricalAccuracy":
                        metrics.append(TfCategoricalAccuracy(metric_class_weights))
                except AttributeError:
                    raise NotImplementedError("the given metric {} is not implemented!".format(metric))

    compile_attributes = train_params["compile_attributes"]

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
        **compile_attributes
    )

    if _log is not None:
        _log.info("model successfully compiled with optimizer %s %s" % (train_params["optimizer"], optimizer))
    else:
        print("model successfully compiled with optimizer %s %s" % (train_params["optimizer"], optimizer))
    return model


def get_class_weights(class_weight, metric_weight, class_distribution):
    """
    Gets the class weights for training and metrics based on class distribution

    :param class_weight: bool, should class weights be considered for training
    :param metric_weight: bool, should class weights be considered for metrics
    :param class_distribution: distribution of classes in dataset
    :return: adjusted class and metric weights or None
    """
    if class_weight:
        class_weights = {i: sum(class_distribution)/class_distribution[i] if class_distribution[i] != 0 else 0
                         for i in range(len(class_distribution))}
        max_weight = np.max([class_weights[k] for k in class_weights])
        class_weights = {k: class_weights[k]/max_weight for k in class_weights}
    else:
        class_weights = None
    if metric_weight:
        metric_class_weights = class_weights
    else:
        metric_class_weights = None
    return class_weights, metric_class_weights
