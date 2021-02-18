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
            #file_write.write(str(redis_get(RedisVar.C_INDEX)))

            file_write.write("\n\nfeature importance calculated on the test data from this side:\n")
            #file_write.write(str(redis_get(RedisVar.FEATURE_IMPORTANCE)))

            file_write.write("\n\nglobal cindex:\n")
            #if redis_get(RedisVar.COORDINATOR):
            #    global_cindex = redis_get(RedisVar.GLOBAL_C_INDEX)
            #    file_write.write(str(global_cindex))

            #if redis_get(RedisVar.CLIENT):
                # TDOD: something is not write here, gives string version of model and not cindex
            #    global_cindex_no_pickle = pickle.loads(redis_get(RedisVar.GLOBAL_C_INDEX_REQUEST))
            #    file_write.write(str(global_cindex_no_pickle))
            file_write.close()

            with open(self.OUTPUT_DIR + '/global_model.pickle', 'wb') as handle:
                jsonpickle.encode(global_rsf)

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
        state_writing_results = 6
        state_finishing = 7

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
                print("[CLIENT] *****************Perform local computation")
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

            #TODO
            if state == state_wait_for_aggregation:
                print("[CLIENT] Wait for aggregation")
                self.progress = 'wait for aggregation'
                if len(self.data_incoming) > 0:
                    print("[CLIENT] Received aggregation data from coordinator.")
                    global_rsf = jsonpickle.decode(self.data_incoming[0])
                    self.data_incoming = []
                    self.client.set_global_rsf(global_rsf)
                    self.global_rsf = global_rsf
                    state = state_writing_results

            # GLOBAL PART

            if state == state_global_aggregation:
                print("[CLIENT] Global computation")
                self.progress = 'computing...'
                if len(self.data_incoming) == len(self.clients):
                    local_rsf_of_all_clients = [jsonpickle.decode(client_data) for client_data in self.data_incoming]
                    self.data_incoming = []
                    aggregated_rsf = self.client.calculate_global_rsf(local_rsf_of_all_clients)
                    #self.client.set_coefs(aggregated_beta)
                    data_to_broadcast = jsonpickle.encode(aggregated_rsf)
                    self.data_outgoing = data_to_broadcast
                    self.status_available = True
                    state = state_writing_results
                    print(f'[CLIENT] Broadcasting computation data to clients', flush=True)

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
