# %%
import pandas as pd
import numpy as np
import seaborn as sns
import sklearn
import os

def read_data(filename):
    data = pd.read_csv(filename, delim_whitespace= True, header= None, index_col= 0)
    data.columns =  [f'A{i}' for i in range(1,16)] + ['B']
    data = data.reset_index(drop = True)
    X = data.iloc[:,:-1]
    Y = data.iloc[:, -1]
    return X, Y


def normalize_and_add_ones(X):
    X_prime = (X - X.min())/ (X.max() - X.min())
    X_prime.insert(0,'A0',1)
    X_prime = np.array(X_prime)
    return X_prime

class RidgeRegression:
    def __init__(self) :
        self.W = None
        
    def fit(self, X_train, Y_train, LAMBDA):
        assert len(X_train.shape) == 2 and \
            X_train.shape[0] == Y_train.shape[0]

        self.W = np.linalg.inv(
            X_train.T.dot(X_train) + 
            LAMBDA * np.identity(X_train.shape[1])
        ).dot(X_train.T).dot(Y_train)
        return self.W

    def predict(self, X_new):
        X_new = np.array(X_new)
        Y_new = X_new.dot(self.W)
        return Y_new

    def compute_RSS(self, Y_new, Y_predicted):
        loss = 1. / Y_new.shape[0] * \
            np.sum((Y_new - Y_predicted) ** 2)
        return loss

    def get_the_best_LAMBDA(self, X_train, Y_train):
        def cross_validation(num_folds, LAMBDA):
            row_ids = np.array(range(X_train.shape[0]))
            divisible =  len(row_ids) - len(row_ids) % num_folds
            valid_ids = np.split(row_ids[:divisible], num_folds)
            valid_ids[-1] = np.append(valid_ids[-1], row_ids[divisible:])
            train_ids = [[j for j in row_ids if j not in valid_ids[i]] for i in range(num_folds)]
            total_RSS = 0
            for i in range(num_folds):
                valid_part = {'X': X_train[valid_ids[i]], 'Y' : Y_train[valid_ids[i]]}
                train_part = {'X' : X_train[train_ids[i]], 'Y' : Y_train[train_ids[i]]}
                self.fit(train_part['X'], train_part['Y'], LAMBDA= LAMBDA)
                Y_predicted = self.predict(valid_part['X'])
                total_RSS += self.compute_RSS(valid_part['Y'], Y_predicted= Y_predicted)
            return total_RSS / num_folds

            
        def range_scan(best_LAMBDA, minium_RSS, LAMBDA_values):
            for current_LAMBDA in LAMBDA_values :
                aver_RSS = cross_validation(num_folds= 5, LAMBDA= current_LAMBDA)
                if aver_RSS < minium_RSS:
                    best_LAMBDA = current_LAMBDA
                    minium_RSS = aver_RSS
            return best_LAMBDA, minium_RSS
        # Search for best lambda from 0 to MAX_LAMBDA - 1
        MAX_LAMBDA = 50
        INIT_RSS = 10 ** 10
        INIT_LAMBDA = 0
        best_LAMBDA, minium_RSS = range_scan(best_LAMBDA= INIT_LAMBDA, minium_RSS= INIT_RSS, LAMBDA_values= range(MAX_LAMBDA))

        LAMBDA_values = [k * 1. / 1000 for k in range(
            max(0, (best_LAMBDA - 1) * 1000),(best_LAMBDA + 1) * 1000,1)]
            
        best_LAMBDA, minium_RSS = range_scan(best_LAMBDA= best_LAMBDA, minium_RSS= minium_RSS, LAMBDA_values= LAMBDA_values)
        return best_LAMBDA


    
if __name__ == '__main__':
  
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, '../datasets/death-rates-data.txt')
    # filename = '../datasets/death-rates-data.txt'
    X, Y = read_data(filename)
    print(X.head())
    X = normalize_and_add_ones(X)
    Y = np.array(Y)
    X_train, Y_train = X[:50], Y[:50]
    X_test, Y_test = X[50:], Y[50:]
    
    ridge_regression = RidgeRegression()
    best_LAMBDA = ridge_regression.get_the_best_LAMBDA(X_train, Y_train)
    print("Best LAMBDA :", best_LAMBDA)
    W_learned = ridge_regression.fit(
        X_train= X_train, Y_train= Y_train, LAMBDA= best_LAMBDA
    )
    Y_predicted = ridge_regression.predict(X_new= X_test)
    print("RSS : ", ridge_regression.compute_RSS(Y_new= Y_test, Y_predicted=Y_predicted))

# %%