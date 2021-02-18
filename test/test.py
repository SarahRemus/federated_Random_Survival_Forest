import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from app.algo import Coordinator, Client

def parse_input(path):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    X = pd.read_csv(path, sep=",").select_dtypes(include=numerics).dropna()
    y = X.loc[:, "TARGET_deathRate"]
    X = X.drop("TARGET_deathRate", axis=1)

    return X, y


class TestLinearRegression(unittest.TestCase):
    def setUp(self):

        X1, y1 = parse_input("client1/client1.csv")
        self.client = Coordinator()

        X2, y2 = parse_input("client2/client2.csv")
        self.client2 = Client()

        xtx, xty = self.client.local_computation(X1, y1)
        xtx2, xty2 = self.client2.local_computation(X2, y2)
        global_coefs = self.client.aggregate_beta([[xtx, xty], [xtx2, xty2]])
        self.client.set_coefs(global_coefs)
        self.client2.set_coefs(global_coefs)

        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        X = pd.read_csv("cancer_reg.csv", sep=",").select_dtypes(include=numerics).dropna()
        y = X.loc[:, "TARGET_deathRate"]
        X = X.drop("TARGET_deathRate", axis=1)
        self.global_model = LinearRegression()
        self.global_model.fit(X, y)

        X = pd.read_csv("client1/client1.csv", sep=",").select_dtypes(include=numerics).dropna()
        self.y_test = X.loc[:, "TARGET_deathRate"]
        self.X_test = X.drop("TARGET_deathRate", axis=1)

    def test_intercept(self):
        print(self.global_model.intercept_)
        print(self.client.intercept_)
        print(self.client2.intercept_)
        np.testing.assert_allclose(self.global_model.intercept_, self.client.intercept_)
        np.testing.assert_allclose(self.global_model.intercept_, self.client2.intercept_)
        np.testing.assert_allclose(self.client2.intercept_, self.client.intercept_)

    def test_coef(self):
        np.testing.assert_allclose(self.global_model.coef_, self.client.coef_, 0.0000001, 0.0000001)
        np.testing.assert_allclose(self.global_model.coef_, self.client2.coef_, 0.00000001, 0.0000001)
        np.testing.assert_allclose(self.client2.coef_, self.client.coef_)

    def test_prediction(self):
        y_pred_global = self.global_model.predict(self.X_test)
        y_pred1 = self.client.predict(self.X_test)
        y_pred2 = self.client2.predict(self.X_test)

        np.testing.assert_allclose(y_pred_global, y_pred1)
        np.testing.assert_allclose(y_pred_global, y_pred2)
        np.testing.assert_allclose(y_pred2, y_pred1)

    def test_score(self):
        score_global = self.global_model.score(self.X_test, self.y_test)
        score1 = self.client.score(self.X_test, self.y_test)
        score2 = self.client2.score(self.X_test, self.y_test)
        np.testing.assert_allclose(score_global, score1)
        np.testing.assert_allclose(score2, score1)
        np.testing.assert_allclose(score2, score_global)


if __name__ == "__main__":
    unittest.main()
