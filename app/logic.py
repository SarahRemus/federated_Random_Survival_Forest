import shutil
import threading
import time

import joblib
import jsonpickle
import pandas as pd
import yaml
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, max_error, mean_absolute_error, \
    mean_absolute_percentage_error, median_absolute_error
from sklearn.model_selection import train_test_split

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

        self.client = None
        self.input = None
        self.sep = None
        self.label_column = None
        self.test_size = None
        self.random_state = None
        self.X = None
        self.y = None
        self.X_test = None
        self.y_test = None
        self.epsilon = None

    def read_config(self):
        with open(self.INPUT_DIR + '/config.yml') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)['fc_linear_regression']
            self.input = config['files']['input']
            self.sep = config['files']['sep']
            self.label_column = config['files']['label_column']
            self.epsilon = config['differential_privacy']['epsilon']
            self.test_size = config['evaluation']['test_size']
            self.random_state = config['evaluation']['random_state']

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
                self.X = pd.read_csv(self.INPUT_DIR + "/" + self.input, sep=self.sep).select_dtypes(
                    include=numerics).dropna()
                self.y = self.X.loc[:, self.label_column]
                self.X = self.X.drop(self.label_column, axis=1)

                if self.test_size is not None:
                    self.X, self.X_test, self.y, self.y_test = train_test_split(self.X, self.y,
                                                                                test_size=self.test_size,
                                                                                random_state=self.random_state)
                state = state_local_computation
            if state == state_local_computation:
                print("[CLIENT] Perform local computation")
                self.progress = 'local computation'
                xtx, xty = self.client.local_computation(self.X, self.y)

                data_to_send = jsonpickle.encode([xtx, xty])

                if self.coordinator:
                    self.data_incoming.append(data_to_send)
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
                    global_coefs = jsonpickle.decode(self.data_incoming[0])
                    self.data_incoming = []
                    self.client.set_coefs(global_coefs)
                    state = state_writing_results

            # GLOBAL PART

            if state == state_global_aggregation:
                print("[CLIENT] Global computation")
                self.progress = 'computing...'
                if len(self.data_incoming) == len(self.clients):
                    data = [jsonpickle.decode(client_data) for client_data in self.data_incoming]
                    self.data_incoming = []
                    aggregated_beta = self.client.aggregate_beta(data)
                    self.client.set_coefs(aggregated_beta)
                    data_to_broadcast = jsonpickle.encode(aggregated_beta)
                    self.data_outgoing = data_to_broadcast
                    self.status_available = True
                    state = state_writing_results
                    print(f'[CLIENT] Broadcasting computation data to clients', flush=True)
            if state == state_writing_results:
                print("[CLIENT] Writing results")
                # now you can save it to a file
                joblib.dump(self.client, self.OUTPUT_DIR + '/model.pkl')
                model = self.client

                if self.test_size is not None:
                    # Make predictions using the testing set
                    y_pred = model.predict(self.X_test)

                    # The mean squared error
                    scores = {
                        "r2_score": [r2_score(self.y_test, y_pred)],
                        "explained_variance_score": [explained_variance_score(self.y_test, y_pred)],
                        "max_error": [max_error(self.y_test, y_pred)],
                        "mean_absolute_error": [mean_absolute_error(self.y_test, y_pred)],
                        "mean_squared_error": [mean_squared_error(self.y_test, y_pred)],
                        "mean_absolute_percentage_error": [mean_absolute_percentage_error(self.y_test, y_pred)],
                        "median_absolute_error": [median_absolute_error(self.y_test, y_pred)]
                    }

                    scores_df = pd.DataFrame.from_dict(scores).T
                    scores_df = scores_df.rename({0: "score"}, axis=1)
                    scores_df.to_csv(self.OUTPUT_DIR + "/scores.csv")

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
