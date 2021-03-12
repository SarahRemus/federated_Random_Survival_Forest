import jsonpickle
import numpy
import sksurv
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
import pandas as pd
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

    def calculate_local_rsf(self, data, data_test, duration_col, event_col):
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
            print("data head")
            print(data.head())
            print("test data head")
            print(data_test.head())
            files = data
            Xt, y, features = bring_data_to_right_format(files, event_col, duration_col)
            X_test, y_test, features = bring_data_to_right_format(data_test, event_col, duration_col)
            # TODO: remove random state later
            # random_state = 20

            # X_train, X_test, y_train, y_test = train_test_split(Xt, y, test_size=0.25)

            rsf = RandomSurvivalForest(n_estimators=1000,
                                       min_samples_split=10,
                                       min_samples_leaf=15,
                                       max_features="sqrt",
                                       n_jobs=-1,
                                       oob_score=True
                                       )
            rsf.fit(Xt, y)



            print("[ALGO]     local rsf: " + str(rsf))

            self.calculate_cindex_and_concordant_pairs(rsf, X_test, y_test)
            # evaluation_on_local_model(rsf, Xt, y, X_test, y_test, y_train)
            return rsf, Xt, y, X_test, y_test, features

    def evaluate_global_model_with_local_test_data(self, global_rsf_pickled, X_test, y_test, feature_names):
        try:
            global_rsf = jsonpickle.decode(global_rsf_pickled)
            cindex = self.calculate_cindex(global_rsf, X_test, y_test)
            feature_importance_as_dataframe = self.calculate_feature_importance(global_rsf, X_test, y_test,
                                                                                feature_names)
            # feature_importance_as_dataframe = pd.DataFrame(['empty'], columns = ['Empty'])
            # brier_score = calculate_integrated_brier_score(global_rsf, Xt, y)
            return cindex, feature_importance_as_dataframe

        except Exception as e:
            print('[ALGO]    evaluate_global_model_with_local_test_data!', e)

    def calculate_feature_importance(self, global_rsf, X_test, y_test, feature_names):
        print("[ALGO]     calculate feature importance")
        # TODO: needs more iterations but is currently taking to long
        perm = PermutationImportance(global_rsf, n_iter=2)
        perm.fit(X_test, y_test)
        feature_importance_as_dataframe = eli5.explain_weights_df(perm, feature_names=feature_names)
        print("\n" + str(feature_importance_as_dataframe))
        return feature_importance_as_dataframe

    def calculate_cindex(self, global_rsf, X_test, y_test):
        cindex = global_rsf.score(X_test, y_test)
        print("[ALGO]     cindex on global model with local test data: " + str(cindex))
        return cindex

    def calculate_cindex_and_concordant_pairs(self, rsf, X_test, y_test):
        print("************************************************************************************")
        print("[TEST C-INDEX] local cindex: " + str(rsf.score(X_test, y_test)))

        prediction_for_test = rsf.predict(X_test)

        event_indicator = [i[0] for i in y_test]

        event_time = [i[1] for i in y_test]

        cindex, concordant, discordant, risk, time = sksurv.metrics.concordance_index_censored(event_indicator, event_time, prediction_for_test)

        print("[TEST C-INDEX] local cindex from prediction: " + str(cindex))
        print("[TEST C-INDEX] concordant pairs: " + str(concordant))
        print("[TEST C-INDEX] discordant pairs: " + str(discordant))


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

    def calculate_global_c_index(self, all_cindeces):
        """
        Calculates the global evaluation of the data of all clients.
        :return: None
        """
        print('[ALGO]     Calculate Global c-index')

        print(f'[ALGO]     all c-indeces: {all_cindeces}')

        mean_c_index = statistics.mean(all_cindeces)

        print("[ALGO]     global cindex: " + str(mean_c_index))
        return mean_c_index


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
