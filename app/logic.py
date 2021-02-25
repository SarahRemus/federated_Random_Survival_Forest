import pickle
import shutil
import threading
import time

import joblib
import jsonpickle
import pandas as pd
import yaml

from app.algo import Coordinator, Client


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
        self.client = None
        self.input = None
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

    def read_config(self):
        with open(self.INPUT_DIR + '/config.yml') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)['fc_rsf']
            self.input = config['files']['input']
            #self.sep = config['files']['sep']
            self.dur_column = config['parameters']['duration_col']
            self.event_column = config['parameters']['event_col']

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

    def write_results(self, global_rsf):
        """
        Writes the results of global_rsf to the output_directory.
        :param global_rsf: Global rsf calculated from the local rsf of the clients
        :param output_dir: String of the output directory. Usually /mnt/output
        :return: None
        """
        try:
            print("[IO]       Write results to output folder:")
            #write_results_for_local_model()
            file_write = open(self.OUTPUT_DIR + '/evaluation_result.txt', 'x')
            file_write.write("Evaluation Results: ")
            file_write.write("\n\nc_index calculated on the test data from this side:\n")
            file_write.write(str(self.cindex_on_global_model))

            #file_write.write("\n\nfeature importance calculated on the test data from this side:\n")
            #file_write.write(str(self.feature_importance_on_global_model))

            file_write.write("\n\nglobal cindex:\n")
            file_write.write(str(self.global_c_index))

            #if self.coordinator:
            #    global_cindex = self.global_c_index
            #    file_write.write(str(global_cindex))

            #if self.client:
            #    global_cindex_no_pickle = self.global_c_index
            #    file_write.write(str(global_cindex_no_pickle))
            file_write.close()



            with open(self.OUTPUT_DIR + '/global_model.pickle', 'wb') as handle:
                pickle.dump(self.global_rsf, handle, protocol=pickle.HIGHEST_PROTOCOL)

            # feature importance
            fi = self.feature_importance_on_global_model
            fi.to_csv(self.OUTPUT_DIR + '/feature_importance.csv')

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

        # === States ===
        state_initializing = 1
        state_read_input = 2
        state_local_computation = 3
        state_wait_for_aggregation = 4
        state_global_aggregation = 5
        state_evaluation_of_global_model = 6
        state_waiting_for_evaluation = 7
        state_aggregation_of_evaluation = 8
        state_writing_results = 9
        state_finishing = 10

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

                numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
                self.data = pd.read_csv(self.INPUT_DIR + "/" + self.input)
                state = state_local_computation

            if state == state_local_computation:
                print("[CLIENT] Perform local computation")
                self.progress = 'local computation'
                rsf, Xt, y, X_test, y_test, features = self.client.calculate_local_rsf(self.data, self.dur_column, self.event_column)

                self.X = Xt
                self.y = y
                self.X_test = X_test
                self.y_test = y_test
                self.features = features

                data_to_send = jsonpickle.encode(rsf)

                if self.coordinator:
                    self.data_incoming.append(data_to_send)
                    print('data incomming coordinator')
                    #print(self.data_incoming)
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
                    ev_result = self.client.evaluate_global_model_with_local_test_data(global_rsf_pickled, self.X_test, self.y_test, self.features)

                if self.client:
                    print('[STATUS]   evaluate global model on local test data CLIENT')
                    global_rsf_pickled = jsonpickle.encode(self.global_rsf)
                    ev_result= self.client.evaluate_global_model_with_local_test_data(global_rsf_pickled, self.X_test,
                                                                                 self.y_test, self.features)

                self.cindex_on_global_model = ev_result[0]
                self.feature_importance_on_global_model = ev_result[1]

                #data_to_send = jsonpickle.encode(ev_result)
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
                    global_cindex_here = jsonpickle.decode(self.data_incoming[0])
                    self.data_incoming = []
                    #self.client.set_global_rsf(global_rsf)
                    self.global_cindex = global_cindex_here
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
                    #self.client.set_coefs(aggregated_beta)
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
                    #print(self.data_incoming)
                    #TODO: something is wrong with the json pickle, should be solved with normal pickle
                    #local_c_of_all_clients = [jsonpickle.decode(client_data) for client_data in self.data_incoming]
                    local_ev_of_all_clients = [pickle.loads(client_data) for client_data in self.data_incoming]
                    local_c_of_all_clients = []
                    for i in local_ev_of_all_clients:
                        local_c_of_all_clients.append(i[0])
                    print("2")
                    print("local of all clients: " + str(local_c_of_all_clients))
                    self.data_incoming = []
                    print("3")
                    aggregated_c = self.client.calculate_global_c_index(local_c_of_all_clients)
                    print("4")
                    self.global_c_index = aggregated_c
                    # self.client.set_coefs(aggregated_beta)
                    data_to_broadcast = jsonpickle.encode(aggregated_c)
                    self.data_outgoing = data_to_broadcast
                    self.status_available = True
                    state = state_writing_results
                    print(f'[CLIENT] Broadcasting EVALUATION data to clients', flush=True)

            if state == state_writing_results:
                print("[CLIENT] Writing results")
                # now you can save it to a file
                #joblib.dump(self.client, self.OUTPUT_DIR + '/model.pkl')
                #model = self.client
                self.write_results(self.global_rsf)
                state = state_finishing

            if state == state_finishing:
                print("[CLIENT] Finishing")
                self.progress = 'finishing...'
                if self.coordinator:
                    time.sleep(10)
                self.status_finished = True
                break

            time.sleep(1)


logic = AppLogic()
