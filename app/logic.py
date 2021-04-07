import pickle
import shutil
import threading
import time

import joblib
import json
import jsonpickle
import pandas as pd
import yaml

from app.algo import Coordinator, Client
from app.app_state import *


class AppLogic:

    def __init__(self):
        # === Status of this app instance ===

        # Indicates whether there is data to share, if True make sure self.data_out is available
        self.status_available = False

        # Only relevant for coordinator, will stop execution when True
        self.status_finished = False

        # === Parameters set during setup ===
        self.id = None
        self.coordinator = None
        self.clients = None

        # === Data ===
        self.data_incoming = []
        self.data_outgoing = None

        # === Internals ===
        self.thread = None
        self.iteration = 0
        self.progress = 'not started yet'

        # === Custom ===
        self.INPUT_DIR = "/mnt/input"
        self.OUTPUT_DIR = "/mnt/output"
        self.data = None
        self.data_test = None
        self.client = None
        self.input = None
        self.input_test = None
        self.dur_column = None
        self.event_column = None
        self.X = None
        self.y = None
        self.X_test = None
        self.y_test = None
        self.features = None
        self.global_rsf = None
        self.cindex_on_global_model = None
        self.feature_importance_on_global_model = None
        self.global_c_index = None
        self.concordant_pairs = None
        self.global_c_index_concordant_pairs = None
        self.actual_concordant_pairs = None
        self.train_samples = None
        self.test_samples = None
        self.random_state = None
        self.time = None

    def read_config(self):
        with open(self.INPUT_DIR + '/config.yml') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)['fc_rsf']
            self.input = config['files']['input']
            self.input_test = config['files']['input_test']
            self.data = pd.read_csv(self.INPUT_DIR + "/" + self.input)
            self.data_test = pd.read_csv(self.INPUT_DIR + "/" + self.input_test)
            print("CONFIG " + str(self.input_test))
            # self.sep = config['files']['sep']
            if self.coordinator:
                self.dur_column = config['parameters']['duration_col']
                self.event_column = config['parameters']['event_col']
                self.random_state = config['parameters']['random_state']

        shutil.copyfile(self.INPUT_DIR + '/config.yml', self.OUTPUT_DIR + '/config.yml')
        print(f'Read config file.', flush=True)

    def handle_setup(self, client_id, coordinator, clients):
        # This method is called once upon startup and contains information about the execution context of this instance
        self.id = client_id
        self.coordinator = coordinator
        self.clients = clients
        print(f'Received setup: {self.id} {self.coordinator} {self.clients}', flush=True)

        self.thread = threading.Thread(target=self.app_flow)
        self.thread.start()

    def handle_incoming(self, data):
        # This method is called when new data arrives
        print("Process incoming data....")
        self.data_incoming.append(data.read())

    def handle_outgoing(self):
        print("Process outgoing data...")
        # This method is called when data is requested
        self.status_available = False
        return self.data_outgoing

    def write_results(self):
        """
        Writes the results of global_rsf to the output_directory.
        :param global_rsf: Global rsf calculated from the local rsf of the clients
        :param output_dir: String of the output directory. Usually /mnt/output
        :return: None
        """
        try:
            print("[IO]       Write results to output folder:")
            file_write = open(self.OUTPUT_DIR + '/evaluation_result.csv', 'x')
            file_write.write("cindex_on_global_model, global_c_index, global_c_index_concordant_pairs, "
                             "training_samples, test_samples, concordant_pairs\n")
            file_write.write(f"{self.cindex_on_global_model},{self.global_c_index},"
                             f"{self.global_c_index_concordant_pairs},{self.train_samples},"
                             f"{self.test_samples},{self.actual_concordant_pairs}")
            file_write.close()

            with open(self.OUTPUT_DIR + '/global_model.pickle', 'wb') as handle:
                pickle.dump(self.global_rsf, handle, protocol=pickle.HIGHEST_PROTOCOL)

            # feature importance
            fi = self.feature_importance_on_global_model
            fi.to_csv(self.OUTPUT_DIR + '/feature_importance.csv', index=False)

        except Exception as e:
            print('[IO]      Could not write result file.', e)
        try:
            file_read = open(self.OUTPUT_DIR + '/evaluation_result.txt', 'r')
            content = file_read.read()
            print(content)
            file_read.close()
        except Exception as e:
            print('[IO]      File could not be read. There might be something wrong.', e)

    def app_flow(self):
        # This method contains a state machine for the client and coordinator instance

        self.time = time.time()
        print(f"START TIME: {self.time}")

        # Initial state
        state = state_initializing
        self.progress = 'initializing...'

        while True:
            if state == state_initializing:
                print("[CLIENT] Initializing")
                if self.id is not None:  # Test if setup has happened already
                    state = state_read_input
                    print("[CLIENT] Coordinator", self.coordinator)
                    if self.coordinator:
                        self.client = Coordinator()
                    else:
                        self.client = Client()

            if state == state_read_input:
                print('[CLIENT] Read input and config')
                self.read_config()
                if self.coordinator:
                    self.data_outgoing = {
                        'event_column': self.event_column,
                        'dur_column': self.dur_column,
                        'random_state': self.random_state,
                    }
                    self.status_available = True
                    state = state_local_computation
                else:
                    state = state_wait_coordinator_input
            if state == state_wait_coordinator_input and not self.coordinator:
                if self.data_incoming:
                    print('[CLIENT] Read input from coordinator')
                    config_data = json.loads(self.data_incoming[0])
                    self.data_incoming = []
                    self.event_column = config_data['event_column']
                    self.dur_column = config_data['dur_column']
                    self.random_state = config_data['random_state']
                    state = state_local_computation

            if state == state_local_computation:
                print("[CLIENT] Perform local computation")
                self.progress = 'local computation'
                rsf, Xt, y, X_test, y_test, features, concordant_pairs, actual_concordant_pairs, train_samples, test_samples = \
                    self.client.calculate_local_rsf(self.data, self.data_test, self.dur_column, self.event_column,
                                                    self.random_state)

                self.X = Xt
                self.y = y
                self.X_test = X_test
                self.y_test = y_test
                self.features = features
                self.concordant_pairs = concordant_pairs
                self.actual_concordant_pairs = actual_concordant_pairs
                self.train_samples = train_samples
                self.test_samples = test_samples
                data_to_send = jsonpickle.encode(rsf)

                if self.coordinator:
                    self.data_incoming.append(data_to_send)
                    print('data incomming coordinator')
                    # print(self.data_incoming)
                    state = state_global_aggregation
                else:
                    self.data_outgoing = data_to_send
                    self.status_available = True
                    state = state_wait_for_aggregation
                    print(f'[CLIENT] Sending computation data to coordinator', flush=True)

            if state == state_wait_for_aggregation:
                print("[CLIENT] Wait for aggregation")
                self.progress = 'wait for aggregation'
                if len(self.data_incoming) > 0:
                    print("[CLIENT] Received aggregation data from coordinator.")
                    global_rsf = jsonpickle.decode(self.data_incoming[0])
                    self.data_incoming = []
                    self.client.set_global_rsf(global_rsf)
                    self.global_rsf = global_rsf
                    state = state_evaluation_of_global_model

            if state == state_evaluation_of_global_model:
                ev_result = []
                if self.coordinator:
                    print('[STATUS]   evaluate global model on local test data COORDINATOR')
                    print('global rsf type: ')
                    print(type(self.global_rsf))
                    global_rsf_pickled = jsonpickle.encode(self.global_rsf)
                    print('could pickle ')
                    ev_result = self.client.evaluate_global_model_with_local_test_data(global_rsf_pickled, self.X_test,
                                                                                       self.y_test, self.features,
                                                                                       self.concordant_pairs)

                if self.client:
                    print('[STATUS]   evaluate global model on local test data CLIENT')
                    global_rsf_pickled = jsonpickle.encode(self.global_rsf)
                    ev_result = self.client.evaluate_global_model_with_local_test_data(global_rsf_pickled, self.X_test,
                                                                                       self.y_test, self.features,
                                                                                       self.concordant_pairs)

                # self.cindex_on_global_model = ev_result[0]
                self.cindex_on_global_model = ev_result[0][0]
                self.feature_importance_on_global_model = ev_result[1]

                # data_to_send = jsonpickle.encode(ev_result)
                data_to_send = pickle.dumps(ev_result)

                if self.coordinator:
                    self.data_incoming.append(data_to_send)
                    state = state_aggregation_of_evaluation
                else:
                    self.data_outgoing = data_to_send
                    self.status_available = True
                    state = state_waiting_for_evaluation
                    print(f'[CLIENT] Sending EVALUATION data to coordinator', flush=True)

            if state == state_waiting_for_evaluation:
                print("[CLIENT] Wait for EVALUATION aggregation")
                self.progress = 'wait for aggregation'
                if len(self.data_incoming) > 0:
                    print("[CLIENT] Received EVALUATION aggregation data from coordinator.")
                    diff_c_index = jsonpickle.decode(self.data_incoming[0])

                    print(f"data_incomming {diff_c_index}")

                    self.global_c_index = diff_c_index[0]
                    self.global_c_index_concordant_pairs = diff_c_index[1]

                    print(f"global cindex: {self.global_c_index}")
                    print(f"global cindex concordant: {self.global_c_index_concordant_pairs}")
                    self.data_incoming = []
                    # self.client.set_global_rsf(global_rsf)
                    state = state_writing_results

            # GLOBAL PART

            if state == state_global_aggregation:
                print("[CLIENT] Global computation")
                self.progress = 'computing...'
                if len(self.data_incoming) == len(self.clients):
                    local_rsf_of_all_clients = [jsonpickle.decode(client_data) for client_data in self.data_incoming]
                    self.data_incoming = []
                    aggregated_rsf = self.client.calculate_global_rsf(local_rsf_of_all_clients)
                    self.global_rsf = aggregated_rsf
                    # self.client.set_coefs(aggregated_beta)
                    data_to_broadcast = jsonpickle.encode(aggregated_rsf)
                    self.data_outgoing = data_to_broadcast
                    self.status_available = True
                    state = state_evaluation_of_global_model
                    print(f'[CLIENT] Broadcasting computation data to clients', flush=True)

            if state == state_aggregation_of_evaluation:
                print("[CLIENT] Global evaluation")
                self.progress = 'evaluating...'
                if len(self.data_incoming) == len(self.clients):
                    print("1")
                    # print(self.data_incoming)
                    # TODO: something is wrong with the json pickle, should be solved with normal pickle
                    # local_c_of_all_clients = [jsonpickle.decode(client_data) for client_data in self.data_incoming]
                    local_ev_of_all_clients = [pickle.loads(client_data) for client_data in self.data_incoming]
                    local_c_of_all_clients = []
                    tuple_c_conc = []
                    for i in local_ev_of_all_clients:
                        local_c_of_all_clients.append(i[0][0])
                        if i[0][1] != 0:
                            tuple_c_conc.append(i[0])
                        else:
                            print("we are not working with this client for evaluation! test set to small!")
                    print("2")
                    print("local of all clients: " + str(local_c_of_all_clients))
                    self.data_incoming = []
                    print("3")
                    aggregated_c = self.client.calculate_global_c_index(local_c_of_all_clients)
                    aggregated_c_with_conc = self.client.calculate_global_c_index_with_concordant_pairs(tuple_c_conc)
                    print('[IMPORTANT!!!!!!!!!!!!!!!!] the aggregated with conc: ' + str(aggregated_c_with_conc))
                    print("4")
                    self.global_c_index = aggregated_c
                    self.global_c_index_concordant_pairs = aggregated_c_with_conc
                    # self.client.set_coefs(aggregated_beta)
                    data_to_broadcast = jsonpickle.encode([aggregated_c, aggregated_c_with_conc])
                    self.data_outgoing = data_to_broadcast
                    self.status_available = True
                    state = state_writing_results
                    print(f'[CLIENT] Broadcasting EVALUATION data to clients', flush=True)

            if state == state_writing_results:
                print("[CLIENT] Writing results")
                # now you can save it to a file
                # joblib.dump(self.client, self.OUTPUT_DIR + '/model.pkl')
                # model = self.client
                self.write_results()

                if self.coordinator:
                    self.data_incoming = ['DONE']
                    state = state_finishing
                else:
                    self.data_outgoing = 'DONE'
                    self.status_available = True
                    break

            if state == state_finishing:
                print("[CLIENT] Finishing")
                self.progress = 'finishing...'
                if len(self.data_incoming) == len(self.clients):
                    self.status_finished = True
                    break

            print("TIME ganz am Ende von TRUE: " + str((time.time()) - self.time))
            time.sleep(1)

        print("TIME ganz am Ende von app flow: " + str((time.time()) - self.time))


logic = AppLogic()

