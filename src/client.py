import os
import json

import numpy as np
import tensorflow as tf

from src.models.model_helpers import m_isup, compile_model, get_class_weights
from src.training_evaluation.train import list_callbacks, setup, training_step, update_standard_metrics, valid_step
from src.dataset_creation.dataset_main import create_dataset
from src.training_evaluation.evaluate_classification import  evaluate_classification_model


class Client:
    def __init__(self, name):
        self.name = name
        self.results_folder = self.setup_folder()

        self.train_data = None
        self.train_class_distribution = None
        self.valid_data = None
    
        self.model = None
        self.callbacks = None
        self.class_weights = None
        self.start_time = None
        self.real_training_time = 0

        self.train_steps_per_epoch = None
        self.train_metrics_list = list()
        self.valid_steps_per_epoch = None
        self.valid_metrics_list = list()
        

    def setup_folder(self):
        '''
        check if folder for storing exists and otherwise create it
        '''
        path = os.path.join('/mnt', 'output', self.name)
        if not os.path.isdir(path):
            os.makedirs(path)
        return path
        
    
    def create_data(self, data_generation_config):
        self.train_data, self.train_class_distribution = create_dataset(data_generation_config,
                                                              usage_mode='train'
                                                            )

        self.valid_data, valid_class_distribution = create_dataset(data_generation_config,
                                                              usage_mode='valid'
                                                            )

        self.train_steps_per_epoch = int(sum(self.train_class_distribution) / data_generation_config['train_batch_size'])
        if self.train_steps_per_epoch == 0:
            self.train_steps_per_epoch = 1
        self.valid_steps_per_epoch = int(sum(valid_class_distribution) / data_generation_config['valid_batch_size'])
        if self.valid_steps_per_epoch == 0:
            self.valid_steps_per_epoch = 1

        print(f'number of records in training dataset: {sum(self.train_class_distribution)}')
        print(f'number of records in validation dataset: {str(sum(valid_class_distribution))}')

        return
    

    def initialise_model(self, model_config, train_config):
        class_weights, metric_class_weights = get_class_weights(train_config['class_weight'],
                                                                train_config['weighted_metrics'],
                                                                self.train_class_distribution
                                                                )

        model = m_isup(model_config)
        self.model = compile_model(model, train_config, metric_class_weights)
        with open(os.path.join(self.results_folder, 'model_json.json'), 'w') as json_file:
            json_file.write(self.model.to_json())
        self.class_weights = class_weights

        return


    def pre_training_callbacks(self, train_config):
        self.epoch = 0
        
        self.callbacks = list_callbacks(train_config)
        # at least, History() should be used as callback
        self.callbacks.append(tf.keras.callbacks.History())
        for callback in self.callbacks:
            callback.set_model(self.model)
            callback.on_train_begin({m.name: m.result() for m in self.model.metrics})
            callback.on_epoch_begin(self.epoch)

        return
    

    def train_local_epoch(self):
        train_loss, self.model = setup(self.model)

        for dpt_idx, datapoint_batch in enumerate(self.train_data.take(self.train_steps_per_epoch)):
            prediction_batch, loss_batch = training_step(self.model, datapoint_batch, self.class_weights, "isup")
            train_loss = train_loss + np.mean(loss_batch)
            update_standard_metrics(self.model, datapoint_batch, prediction_batch)
     
        train_metrics = {**{m.name: m.result() for m in self.model.compiled_metrics._metrics}, 'loss': train_loss/(dpt_idx+1)}
        print('Training -   ')
        for v in train_metrics:
            print('    {:s}: {:.4f}   '.format(v, train_metrics[v]))
        self.train_metrics_list.append(train_metrics)
        return
    

    def aggregate_weights(self, list_of_weights, method):
        weights = list([list() for _ in range(len(list_of_weights[0]))])
        if method == 'fed_avg':
            for weight in list_of_weights:
                for i, w in enumerate(weight):
                    weights[i].append(w)
            final_weights = [np.array(sum(w_l) / len(list_of_weights)) for w_l in weights]
        else:
            exit(f"Aggregation method {method} is not implemented.")

        return final_weights
        

    def validate_local_epoch(self):
        valid_loss, self.model = setup(self.model)

        for dpt_idx, datapoint_batch in enumerate(self.valid_data.take(self.valid_steps_per_epoch)):
            prediction_batch, loss_batch = valid_step(self.model, datapoint_batch, self.class_weights, "isup")
            valid_loss += np.mean(loss_batch)
            update_standard_metrics(self.model, datapoint_batch, prediction_batch)

        valid_metrics = {**{'_'.join(('val', m.name)): m.result() for m in self.model.compiled_metrics._metrics}, 'val_loss': valid_loss/(dpt_idx+1)}
        for callback in self.callbacks:
            callback.on_epoch_end((self.epoch, self.results_folder), {**self.train_metrics_list[-1], **valid_metrics})
        print('Validation -   ')
        for v in valid_metrics:
            print('    {:s}: {:.4f}   '.format(v, valid_metrics[v]))
        print('')
        self.valid_metrics_list.append(valid_metrics)
        return

  
    def get_monitor_values(self, monitor_val):
        values = self.callbacks[-1].history[monitor_val]
        return values


    def test_final_model(self, data_generation_config, evaluation_config, best_epoch):
        best_model_path = [f for f in os.listdir(self.results_folder) if 'weights' in f and int(f.split('weights')[1].split('_')[0]) == best_epoch][0]
        self.model.load_weights(os.path.abspath(os.path.join(self.results_folder, best_model_path)))
        os.rename(os.path.join(self.results_folder, best_model_path),
                  os.path.join(self.results_folder, best_model_path.replace('temp', 'best')))
        files = [f for f in os.listdir(os.path.abspath(self.results_folder)) if 'weights' in f]
        for f in files:
            if 'best' not in f:
                os.remove(os.path.join(self.results_folder, f))

        test_data, test_class_distribution = create_dataset(data_generation_config=data_generation_config,
                                                            usage_mode='test'
                                                            )        
    
        test_results = evaluate_classification_model(self.model, test_data, test_class_distribution,
                                                     "isup", evaluation_config['metrics'],
                                                     self.results_folder)

        # STORE RESULTS
        train_history = {}
        for k, v in self.callbacks[-1].history.items():
            train_history[k] = np.array(v).astype(float).tolist()

        allresults = {'train_results': train_history, 'test_results': test_results}
        with open(os.path.join(self.results_folder, 'results.json'), 'w') as resultsfile:
            json.dump(allresults, resultsfile)
