import numpy as np
import numpy
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
from sksurv.ensemble import RandomSurvivalForest
import eli5
from eli5.sklearn import PermutationImportance
import statistics
from sksurv.metrics import integrated_brier_score
import matplotlib.pyplot as plt


class Client:

    global_rsf_ = None

    def set_global_rsf(self, global_rsf):
        self.global_rsf_ = global_rsf[0]


    def calculate_local_rsf(self, data, duration_col, event_col):
        """
        Calculate the local rsf of a client
        :return: the local rsf
        """

        if data is None:
            print('[ALGO]     No data available')
            return None
        else:
            # TODO: currently hard coded, should be input
            print("[ALGO]     calculate local rsf")
            files = data
            Xt, y, features = bring_data_to_right_format(files, event_col, duration_col)
            # TODO: remove random state later
            random_state = 20

            X_train, X_test, y_train, y_test = train_test_split(
                Xt, y, test_size=0.25, random_state=random_state)

            rsf = RandomSurvivalForest(n_estimators=1000,
                                       min_samples_split=10,
                                       min_samples_leaf=15,
                                       max_features="sqrt",
                                       n_jobs=-1,
                                       oob_score=True
                                       )
            rsf.fit(X_train, y_train)
            print("[ALGO]     local rsf: " + str(rsf))
            #evaluation_on_local_model(rsf, Xt, y, X_test, y_test, y_train)
            return rsf, Xt, y, X_test, y_test, features



class Coordinator(Client):

    def calculate_global_rsf(self, locally_trained_forests):
        """
        Calculates the global rsf of the data of all clients.
        :return: None
        """
        print('[ALGO]     Calculate Global RSF')
        print('[ALGO]     length locally_trained_forests ' + str(len(locally_trained_forests)))
        firstTree = locally_trained_forests[0]
        remaining_trees = locally_trained_forests.copy()
        remaining_trees.remove(firstTree)

        for forest in remaining_trees:
            firstTree.estimators_ = firstTree.estimators_ + forest.estimators_
            firstTree.n_estimators = len(firstTree.estimators_)
        global_rsf = firstTree
        print(global_rsf)
        return global_rsf

def bring_data_to_right_format(data, event, time):
    # read data and reformat so sckit-survival can work with it

    df = data
    # Assign True/ False variables as event-occurs Data, not 1/0
    df[event] = df[event].astype('bool')

    # create nparray with event and time data as tuples
    y = df[[event, time]].copy()
    s = y.dtypes
    y_finished = np.array([tuple(x) for x in y.values], dtype=list(zip(s.index, s)))

    headers = data.columns.values.tolist()
    headers.remove(event)
    headers.remove(time)

    # create nparray with the features values
    x = df[headers].copy()
    x_finished = x.to_numpy().astype('float64')
    features = headers

    return x_finished, y_finished, features