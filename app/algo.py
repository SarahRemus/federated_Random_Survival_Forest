import numpy as np
from sklearn.linear_model import LinearRegression


class Client(LinearRegression):

    coef_ = None
    intercept_ = None

    def set_coefs(self, coef):
        self.coef_ = coef[1:]
        self.intercept_ = coef[0]

    def local_computation(self, X, y, eps=None):

        def add_dp(org_matrix, epsilon):
            if eps:
                sensitivity = np.max(np.sum(np.abs(org_matrix), axis=0))
                mean = 0
                lambda_laplace = sensitivity / epsilon
                noise_matrix = np.random.laplace(mean, lambda_laplace, size=org_matrix.shape)

                return org_matrix + noise_matrix
            else:
                return org_matrix

        column_one = np.ones((X.shape[0], 1)).astype(np.uint8)
        X = np.concatenate((column_one, X), axis=1)
        XT_X_matrix = add_dp(np.dot(X.T, X), epsilon=eps)
        XT_y_vector = add_dp(np.dot(X.T, y), epsilon=eps)

        return XT_X_matrix, XT_y_vector

    def predict(self, X):
        return np.dot(X, self.coef_) + self.intercept_


class Coordinator(Client):

    def aggregate_matrices_(self, matrices):
        matrix = matrices[0]
        for i in range(1, len(matrices)):
            matrix = np.add(matrix, matrices[i])
        matrix_global = matrix

        return matrix_global

    def aggregate_beta(self, local_results):
        XT_X_matrices = [client[0] for client in local_results]
        XT_X_matrix_global = self.aggregate_matrices_(XT_X_matrices)

        XT_y_vectors = [client[1] for client in local_results]
        XT_y_vector_global = self.aggregate_matrices_(XT_y_vectors)

        XT_X_matrix_inverse = np.linalg.inv(XT_X_matrix_global)
        beta_vector = np.dot(XT_X_matrix_inverse, XT_y_vector_global)

        return beta_vector
