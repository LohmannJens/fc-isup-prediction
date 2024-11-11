import os
import logging

import numpy as np
import tensorflow as tf

from src.dataset_creation.dataset_processing import tf_read_image
from src.dataset_creation.image_transformations.data_augmentation import tf_augment_image
from src.dataset_creation.image_transformations.normalization import tf_normalize_image
from src.dataset_creation.read_from_csv import create_img_and_label_series
from src.dataset_creation.convert_to_tf_dataset import make_image_dataset
from src.dataset_creation.label_encoding import transform_labels

AUTOTUNE = tf.data.experimental.AUTOTUNE


def create_dataset(data_generation_config={}, usage_mode='train'):
    """
    Read images and labels from .csv file, where the image paths are stored in column 'img_path'
    and the labels are stored in data_generation_config['annotation_column']

    :param data_generation_config: a dictionary with
                                                  train_csv_file (+test and valid): filename with path to csv
                                                  batch_size: batch size used for training and validation
                                                  directory: directory where the images are stored (/data)
    :param usage_mode: train, test or valid
    :return: full dataset, distribution of classes
    """
    directory = os.path.join('/mnt', 'input')

    if usage_mode == 'train':
        csv_filepath = os.path.join(directory, 'train.csv')
        batch_size = data_generation_config['train_batch_size']
    elif usage_mode == 'valid':
        csv_filepath = os.path.join(directory, 'valid.csv')
        batch_size = data_generation_config['valid_batch_size']
    elif usage_mode == 'test':
        csv_filepath = os.path.join(directory, 'test.csv')
        batch_size = 1
    else:
        raise NameError('UnknownUsageModeDataGenerator')

    data_generator, class_distribution = dataset_from_csv(data_generation_config,
                                                          csv_filepath,
                                                          directory,
                                                          batch_size,
                                                          usage_mode
                                                          )
    return data_generator, class_distribution


def dataset_from_csv(data_generation_config, csv_filepath, image_directory, batch_size=0, usage_mode='train', _log=logging):
    """
    Create a tf.data.Dataset from a csv file which contains paths to images and their labels                           

    :param data_generation_config: a dictionary with at least
                                directory: string where the images are stored
                                annotation_column : string name of column in csv where the class label is to be found
                                label_type: string e.g. gleason, isup - how to map label to integer
                                additional_columns: list ['name','use_as'] additional columns of csv to be used as input or label
                                number_of_classes: how many classes/intervals we have
                                seed: can be None but should be set for reproducibility
    :param csv_filepath: full path to csv file
    :param image_directory: path to images
    :param batch_size: training batch size
    :param usage_mode: 'train', 'valid' or 'test'
    :param _log: logger
    :return: dataset, distribution of classes
    """

    if data_generation_config['seed'] is not None:
        np.random.seed(data_generation_config['seed'])
        tf.random.set_seed(data_generation_config['seed'])

    # read image paths, labels and masks from csv as they are
    images, labels, additional_columns = create_img_and_label_series(csv_filepath, data_generation_config)

    # encode the labels as wanted
    transformed_labels, array_labels, interval_limits = transform_labels(np.array(labels))

    # transform the image paths and labels to a dataset
    dataset = make_image_dataset(images, transformed_labels, additional_columns)

    # now read the images from the paths (keep path in dataset for evaluation or inspection later)
    channels = 3
    dataset = dataset.map(lambda x: ({'images': tf_read_image(x['image_paths'],
                                                              image_directory,
                                                              channels,
                                                              ),
                                      **{key: x[key] for key in x}}),
                          num_parallel_calls=AUTOTUNE)

    _log.warning('CAUTION: if image is not cut into patches, all images need to have the same shape. '
                     'This is hard coded to be 2048 in dataset_main.py!')
    dataset = dataset.map(lambda x: ({'images': tf.reshape(x['images'], [2048, 2048, channels]),
                                      **{key: x[key] for key in x if key not in ['images']}}),
                          num_parallel_calls=AUTOTUNE)

    # resize images to match the final model input shape (mostly for memory reasons)
    image_size = data_generation_config['resize']
    dataset = dataset.map(lambda x: ({'images': tf.image.resize(x['images'], [image_size, image_size]),
                                      **{key: x[key] for key in x if key not in ['images']}}),
                          num_parallel_calls=AUTOTUNE)

    # for survival analysis, it is best we keep the interval limits written down in the dataset
    if interval_limits is not None:
        dataset = dataset.map(lambda x: ({**{key: x[key] for key in x},
                                          'interval_limits': tf.convert_to_tensor(interval_limits)}),
                              num_parallel_calls=AUTOTUNE)
    # in case not only an image is used as input and the label is used as output:
    # output for example for dataset['images']: tuple(patches, whole_image, time_intervals, psa_value)
    # output for exmaple for dataset['label']: tuple(event_month, censored)
    dataset, additional_inputs = adjust_dataset_to_model(dataset)

    # cache the dataset in memory to save computational time
    dataset = dataset.cache()

    class_distribution = np.array([sum(array_labels == c) for c in range(6)])

    # in case of training, the dataset needs to be shuffled and can be augmented
    if usage_mode == 'train':
        dataset = dataset.shuffle(tf.cast(tf.reduce_min((500, tf.reduce_sum(class_distribution))), 'int64'), reshuffle_each_iteration=True)
        random_augmentation = True
        augmentation_config = dict({"rotation": 90.0, "width_shift": 0.1, "height_shift": 0.1, "brightness": 0.05,
                                    "horizontal_flip": True, "vertical_flip": True, "fill_mode": "constant", "cval": 255})
            
        dataset = dataset.map(lambda x: ({'images': tuple([tf_augment(x['images'][i], additional_inputs[i],
                                                                      random_augmentation,
                                                                      **augmentation_config) for i in range(len(x['images']))]),
                                          **{key: x[key] for key in x if key not in ['images']}}),
                              num_parallel_calls=1)

    dataset = dataset.map(lambda x: ({'images': tf_normalize(x['images'][0], channels, 'images'),
                                      **{key: x[key] for key in x if key not in ['images']}}),
                          num_parallel_calls=AUTOTUNE)

    # produce batches
    dataset = dataset.batch(batch_size, drop_remainder=False)
    if usage_mode != 'test':
       dataset = dataset.repeat()
    dataset = dataset.prefetch(3)
    return dataset, class_distribution


def adjust_dataset_to_model(dataset):
    """
    In case multiple inputs than just a single image
    and multiple labels than just a single label are needed, this function is used
    For a dataset with 'images' it adds additional information, so that e.g. when adding 'age'
    before: dataset['images']: np array of batch_size, height, width, n_channels
    after: dataset['images']: list(np array image, np array age)

    For a dataset with 'censored' as additional label:
    before: dataset['labels']: np array of batch_size, n_intervals
    after:  dataset['labels']: np array of batch_size, n_intervals+1
            (this is different from input, because it is handled differently in the loss)

    :param dataset: tf.data.Dataset (with 'images' and 'labels' as input and labels, plus other entries as additional info
    :return: tf.data.Dataset with 'images' as input values and 'labels' as labels used for loss function
    """
    additional_inputs = list()
    additional_labels = list()
    additional_inputs.insert(0, 'images')

    dataset = dataset.map(lambda x: {   # inputs with single value need an additional dimension (1x1)
                                     **{'_'.join((a, 'exp')): tf.cast(
                                                tf.expand_dims(x[a], -1),
                                                'float32')
                                                for a in additional_inputs if (a not in ['images', 'img_original', 'interval_limits'])},
                                     # inputs that have one value per time step or are images (without patches)
                                     **{'_'.join((a, 'exp')): tf.cast(
                                                                     x[a],
                                                                     'float32')
                                                                     for a in additional_inputs if
                                                                     (a in ['interval_limits', 'img_original'])},
                                     # inputs that have 4 dimensions already n_patches x w x h x channels
                                     **{'_'.join((a, 'exp')): x[a] for a in additional_inputs if (a in ['images'])},
                                     # the rest is just additional information that is not used as network input
                                     **{key: x[key] for key in x if key not in ['images', 'img_original']}})

    dataset = dataset.map(lambda x: {'images': tuple(x['_'.join((a, 'exp'))] for a in additional_inputs),
                                         **{key: x[key] for key in x if key not in ['images', 'images_exp',
                                                                                    'img_original_exp',
                                                                                    'interval_limits_exp',
                                                                                    'img_original']}})

    for add_labels in additional_labels:
        dataset = dataset.map(lambda x: {'labels': tf.concat((x['labels'],
                                                              tf.cast(tf.expand_dims(x[add_labels], 0), 'float32')
                                                              ), 0),
                                         **{key: x[key] for key in x if key not in ['labels']}})

    return dataset, additional_inputs


def tf_normalize(image, image_channels, input_type):
    """
    wrapper that can distinguish between normalizing images and other inputs
    """

    if input_type in ['images', 'img_original']:
        return tf_normalize_image(image, image_channels)
    else:
        return image


@tf.function
def tf_augment(image, input_type, random_augmentation, **augmentation_config):
    """
    wrapper for augmentation
    """
    if input_type in ['img_original', 'images']:
        if len(image.shape) == 3:
            return tf_augment_image(tf.expand_dims(image, 0), random_augmentation, **augmentation_config)[0]
        else:
            return tf_augment_image(image, random_augmentation, **augmentation_config)

    else:
        return image
