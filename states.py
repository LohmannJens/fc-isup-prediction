import os
import yaml
import time

import tensorflow as tf

from FeatureCloud.app.engine.app import AppState, app_state, Role

from src.client import Client


@app_state('initial')
class InitialState(AppState):

    def register(self):
        self.register_transition('distribute_model', Role.COORDINATOR)
        self.register_transition('pre_training', Role.PARTICIPANT)


    def run(self):
        # define threads used
        tf.config.threading.set_inter_op_parallelism_threads(2)
        tf.config.threading.set_intra_op_parallelism_threads(2)

        # read in the config
        config_file = os.path.join(os.getcwd(), "mnt", "input", "config.yaml")
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)

        # initialize client
        client = Client(self.id)
        
        # load images
        client.create_data(config['data_generation'])

        # initialise model
        client.initialise_model(config['model'], config['training'])

        # store client class and config
        self.store(key='client', value=client)
        self.store(key='config', value=config)
        
        if self.is_coordinator:
            return 'distribute_model'
        return 'pre_training'
    

@app_state('distribute_model')
class DistributeModelState(AppState):

    def register(self):
        self.register_transition('pre_training', Role.COORDINATOR)


    def run(self):
        client = self.load('client')

        # send initialized model of coordinator to all clients
        self.broadcast_data(client.model.get_weights(), send_to_self=True)

        return 'pre_training'


@app_state('pre_training')
class PreTrainingState(AppState):

    def register(self):
        self.register_transition('train', Role.BOTH)


    def run(self):
        client = self.load('client')
        config = self.load('config')

        # receive initial model by the server
        new_weights = self.await_data(n=1, is_json=False)
        client.model.set_weights(new_weights)

        # list and run callbacks
        client.pre_training_callbacks(config['training'])
        client.start_time = time.time()

        # store client class and config
        self.store(key='client', value=client)

        return 'train'


@app_state('train')
class TrainState(AppState):

    def register(self):
        self.register_transition('aggregate', Role.COORDINATOR)
        self.register_transition('validate', Role.PARTICIPANT)


    def run(self):
        client = self.load('client')
        config = self.load('config')

        # run callbacks
        epoch_start_time = time.time()
        for callback in client.callbacks:
            callback.on_epoch_begin(client.epoch)

        # run training for n epochs on each client
        for _ in range(config['training']['local_epochs']):
            client.train_local_epoch()
    
        # send model weights for aggregation into central model
        self.send_data_to_coordinator(client.model.weights, send_to_self=True, use_smpc=False)
        client.real_training_time += (time.time() - epoch_start_time)
        self.store(key='client', value=client)

        if self.is_coordinator:
            return 'aggregate'
        return 'validate'


@app_state('aggregate')
class AggregateState(AppState):

    def register(self):
        self.register_transition('validate', Role.COORDINATOR)


    def run(self):
        client = self.load('client')
        config = self.load('config')

        # await weights of clients
        list_of_weights = self.gather_data(is_json=False)

        # aggregate the weights of the clients by basic fed_avg
        new_weights = client.aggregate_weights(list_of_weights, config['training']['aggregation_method'])

        # distribute aggregated model to the clients
        self.broadcast_data(new_weights, send_to_self=True)

        return 'validate'


@app_state('validate')
class ValidateState(AppState):

    def register(self):
        self.register_transition('train', Role.BOTH)
        self.register_transition('test', Role.PARTICIPANT)
        self.register_transition('select_best_epoch', Role.COORDINATOR)


    def run(self):
        client = self.load('client')
        config = self.load('config')

        # receive aggregated model by the server
        new_weights = self.await_data(n=1, is_json=False)
        client.model.set_weights(new_weights)

        # start validation routine based on parameters given in config
        client.validate_local_epoch()
        client.epoch = client.epoch + 1

        self.store(key='client', value=client)

        # if training is not finished based on number of epochs
        if client.epoch != config['training']['epochs']:
            return 'train'
        # training is finished, select best model and final testing
        else:
            print(f'Overall time for training: {round(time.time()-client.start_time, 0)} seconds')
            print(f'Real training time: {round(client.real_training_time, 0)} seconds')
            
            values = client.get_monitor_values(config['training']['monitor_val'])
            self.send_data_to_coordinator(values, send_to_self=True, use_smpc=False)
            
            if self.is_coordinator:
                return 'select_best_epoch'
            return 'test'


@app_state('select_best_epoch')
class SelectBestEpochState(AppState):

    def register(self):
        self.register_transition('test', Role.COORDINATOR)


    def run(self):
        # get monitored values per client
        list_of_metrics = self.gather_data(is_json=False)

        # average over all clients and get highest value
        mean_tensor = tf.reduce_mean(tf.stack(list_of_metrics), axis=0)
        max_index = tf.argmax(mean_tensor).numpy()
        print(f"Best epoch:\t{max_index}")

        # send best epoch to all clients
        self.broadcast_data(max_index, send_to_self=True)

        return 'test'


@app_state('test')
class TestState(AppState):

    def register(self):
        self.register_transition('terminal', Role.BOTH)  

    def run(self):
        client = self.load('client')
        config = self.load('config')

        # get best epoch from coordinator
        best_epoch = self.await_data(n=1, is_json=False)

        # start final evaluation of the model
        client.test_final_model(config['data_generation'], config['evaluation'], best_epoch)

        return 'terminal'
        