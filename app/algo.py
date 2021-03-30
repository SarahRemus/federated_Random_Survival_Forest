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

    def calculate_local_rsf(self, data, data_test, duration_col, event_col, random_state):
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
                                       random_state=random_state,
                                       oob_score=True
                                       )
            rsf.fit(Xt, y)



            print("[ALGO]     local rsf: " + str(rsf))
            train_samples = len(Xt)
            test_samples = len(X_test)

            if len(y_test) == 0:
                print("[NOT JUSING TEST SET!] it is empty")
                return rsf, Xt, y, X_test, y_test, features, 0, 0, train_samples, test_samples
            else:
                try:
                    concordant_pairs = self.calculate_cindex_and_concordant_pairs(rsf, X_test, y_test)
                    actual_concordant_pairs = concordant_pairs
                    print("CONCORDANT BEVOR ANYTHING:  " + str(concordant_pairs))


                    #TODO: really random, think about number that makes sense
                    if concordant_pairs > 20:
                        print("[JUSING TEST SET!] concordant pairs: " + str(concordant_pairs) + " are more than 20")
                        return rsf, Xt, y, X_test, y_test, features, concordant_pairs, actual_concordant_pairs, train_samples, test_samples
                    else:
                        print("[NOT JUSING TEST SET!] concordant pairs: " + str(concordant_pairs) + " are less than 20")
                        #rsf, Xt, y, X_test, y_test, features, concordant_pairs = \
                            #self.handle_to_small_test_set(data, data_test, duration_col, event_col, random_state)
                        #not merging
                        #return rsf, Xt, y, X_test, y_test, features, concordant_pairs, actual_concordant_pairs, train_samples, test_samples
                        return rsf, Xt, y, X_test, y_test, features, 0, actual_concordant_pairs, train_samples, test_samples

                except ValueError as e:
                    print("ERROR: " + str(e))
                    return rsf, Xt, y, X_test, y_test, features, 0,0, train_samples, test_samples

    def evaluate_global_model_with_local_test_data(self, global_rsf_pickled, X_test, y_test, feature_names, concordant_pairs):
        try:
            print("[EVALUATION!!!] CONCORDANT PAIRS: "+ str(concordant_pairs))
            if concordant_pairs != 0:
                global_rsf = jsonpickle.decode(global_rsf_pickled)
                cindex = self.calculate_cindex(global_rsf, X_test, y_test)
                feature_importance_as_dataframe = self.calculate_feature_importance(global_rsf, X_test, y_test,
                                                                                    feature_names)

                # feature_importance_as_dataframe = pd.DataFrame(['empty'], columns = ['Empty'])
                # brier_score = calculate_integrated_brier_score(global_rsf, Xt, y)
                return (cindex, concordant_pairs), feature_importance_as_dataframe
            else:
                return (0, 0), None

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
        print("X_test")
        print(X_test)
        print("y_test")
        print(y_test)

        print("[TEST C-INDEX] local cindex: " + str(rsf.score(X_test, y_test)))

        prediction_for_test = rsf.predict(X_test)

        event_indicator = [i[0] for i in y_test]

        event_time = [i[1] for i in y_test]

        result = []

        #cindex, concordant, discordant, risk, time = sksurv.metrics.concordance_index_censored(event_indicator, event_time, prediction_for_test)
        result = sksurv.metrics.concordance_index_censored(event_indicator, event_time, prediction_for_test)
        print("[TEST C-INDEX] result: " + str(result))
        print("[TEST C-INDEX] local cindex from prediction: " + str(result[0]))
        print("[TEST C-INDEX] concordant pairs: " + str(result[1]))
        print("[TEST C-INDEX] discordant pairs: " + str(result[2]))
        return result[1]

    def handle_to_small_test_set(self, data, data_test, duration_col, event_col, random_state):
        print("[INFO!!!!] we are handling a too small test set")
        concat_data = [data, data_test]
        new_training = pd.concat(concat_data)
        Xt, y, features = bring_data_to_right_format(new_training, event_col, duration_col)
        X_test, y_test, features = bring_data_to_right_format(data_test, event_col, duration_col)
        rsf = RandomSurvivalForest(n_estimators=1000,
                                   min_samples_split=10,
                                   min_samples_leaf=15,
                                   max_features="sqrt",
                                   n_jobs=-1,
                                   random_state = random_state,
                                   oob_score=True
                                   )
        rsf.fit(Xt, y)

        print("[ALGO]     local rsf: " + str(rsf))
        return rsf, Xt, y, X_test, y_test, features, 0













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

        #if 0 in all_cindeces:
        #    all_cindeces.remove(0)

        all_cindeces = list(filter((0).__ne__, all_cindeces))

        print(f'[ALGO]  after   all c-indeces: {all_cindeces}')

        mean_c_index = statistics.mean(all_cindeces)

        mean_test = sum(all_cindeces)/len(all_cindeces)
        print("[ALGO]     global cindex test: " + str(mean_test))

        print("[ALGO]     global cindex: " + str(mean_c_index))
        return mean_c_index

    def calculate_global_c_index_with_concordant_pairs(self, all_cindeces_and_con_pairs):
        all_conc = sum([i[1] for i in all_cindeces_and_con_pairs])
        con_mul_c = sum([i[1]*i[0] for i in all_cindeces_and_con_pairs])

        if all_conc != 0:
            result = con_mul_c/all_conc
            print("[TEST C-INDEX WITH CONC] all_conc = " + str(all_conc))
            print("[TEST C-INDEX WITH CONC] con_mul_c = " + str(con_mul_c))
            print("[TEST C-INDEX WITH CONC] result = " + str(result))
            return result
        else:
            print("[C Index computation failed] all test sets where too small = " + str(all_conc))
            return 0

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
