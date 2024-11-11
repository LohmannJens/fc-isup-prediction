# Federated ISUP grade prediction FeatureCloud App

## Description
A FeatureCloud app to predict ISUP grades [1] from tissue microarray (TMA) images.
It was developed based on the first step of eCaReNet [2] for prostate cancer relapse
over time prediction.


## Input
In each client the following folders and files must be given:
- `images`: folder that contains the TMA images as PNG
- `train.csv`: local training data, needs to include image path and ISUP grade
- `valid.csv`: local validation data, needs to include image path and ISUP grade
- `test.csv`: local test data, needs to include image path and ISUP grade

Over all clients these files are the same:
- `config.yaml`: config file with information about the trainig
- `InceptionV3.hdf`: Inception-v3 model file adapted to right image size, pretrained on ImageNet data

The input images should be pre-processed as described in the eCaReNet paper [2].


## Output
- `results.csv`: predicted class per image and metadata for the test dataset
- `confusion_matrix.png`: confusion matrix of the test dataset predictions (absolute values)
- `confusion_matrix_relative.png`: confusion matrix of the test dataset predictions (relative values)
- `model_json.json`: model architecture, necesarry to load the model weights in tensorflow
- `best_weights{epoch}_accur{value}.hdf`: weights of the best performing model, filename includes best epoch {epoch} and corresponding validation accuracy {value}
- `results.json`: model metrics during training and final evaluation


## Workflows
Combinations with other FeatureCloud apps was not tested yet.


## Config
Use the config file to customize your training and evaluation.
Needs to be uploaded together with the training data as `config.yaml`
```
data_generation:
  train_batch_size: 2 # int, batch size for training
  valid_batch_size: 2 # int, batch size for validation 
  drop_cases: # list of lists
    - ['-1', 'isup'] # define value to drop, define column where to drop from
  resize: 256 # int, needs to be according to the Inception-v3 model that is used
  seed: 161 # int, definition of a seed to make results reproducible

# ------  ------  ------  ------  ------  ------  ------

model:
  additional_input: [] # list, column names of additional inputs
  dense_layer_nodes: [32] # list, number of nodes for dense layer
  keras_model_params: # dict, parameter input for the keras model
    weights: imagenet # None or 'imagenet'
    input_shape: [256, 256, 3]
  cut_off_layer: 'mixed4_base' # str, layer where Inception-v3 is cut off to add eCaReNet layers

# ------  ------  ------  ------  ------  ------  ------

training:
  epochs: 3 # int, how many epochs to train (number of communications)
  local_epochs: 1 # int, how many epochs to train per communication round (before merging models again)
  monitor_val: 'val_tf_categorical_accuracy' # str, metric that is used to determine best epoch
  callbacks: # dict, new callbacks like lr scheduler or early stopping can be defined
  optimizer: # dict
    name: SGD # str, tf.keras optimizer, e.g. 'Adam', 'Nadam'
    params: # additional parameters as dict
      learning_rate: 0.005
  loss_fn: isup_cce_loss # str, loss function for the optimizer
  compile_metrics: # list, metrics for the optimizer
    - TfCategoricalAccuracy
  compile_attributes: {} # dict, additional attributes 
  class_weight: True # bool, weighting classes by occurrence
  weighted_metrics: False # bool, using weighted metrics
  aggregation_method: fed_avg # str, defines which aggregation method to use

# ------  ------  ------  ------  ------  ------  ------

evaluation:
  metrics: # list, lists the metrics to evaluate the test dataset 
    - accuracy
    - f1_score
    - kappa
```

## Test data
For testing the functionality of the app we provide dummy data in `data/testdata`.
The dummy test data includes two clients with quadratic images (2048 x 2048).
Additionally, the folder includes a pretrained model stored as `InceptionV3.hdf`.
The model was configured to expect images of size 256 x 256 pixels as input.


## References
[1] Egevad, L., Delahunt, B., Srigley, J. R., & Samaratunga, H. (2016). International Society of Urological Pathology (ISUP) grading of prostate cancer – An ISUP consensus on contemporary grading. APMIS, 124(6), 433–435. https://doi.org/10.1111/apm.12533

[2] Dietrich, E., Fuhlert, P., Ernst, A., Sauter, G., Lennartz, M., Stiehl, H. S., Zimmermann, M., & Bonn, S. (2021). Towards Explainable End-to-End Prostate Cancer Relapse Prediction from H&E Images Combining Self-Attention Multiple Instance Learning with a Recurrent Neural Network. In S. Roy, S. Pfohl, E. Rocheteau, G. A. Tadesse, L. Oala, F. Falck, Y. Zhou, L. Shen, G. Zamzmi, P. Mugambi, A. Zirikly, M. B. A. McDermott, & E. Alsentzer (Eds.), Proceedings of Machine Learning for Health (Vol. 158, pp. 38–53). PMLR. https://proceedings.mlr.press/v158/dietrich21a.html
