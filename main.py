from math import log10
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


def process_data(dataset):
    dataset['alpha'] = dataset['alpha'].apply(lambda x: log10(x))
    dataset.drop(['id'], axis=1, inplace=True)
    dataset = pd.concat([dataset, pd.get_dummies(dataset['penalty'])], axis=1)
    dataset.drop(['penalty'], axis=1, inplace=True)
    max_jobs = dataset['n_jobs'].max()
    dataset['n_jobs'] = dataset['n_jobs'].apply(lambda x: max_jobs if x == -1 else x)
    return dataset


def to_log_label(origin_label):
    return pd.Series(origin_label).apply(lambda x: log10(x))


def resume_from_log_label(encoded_label):
    return pd.Series(encoded_label).apply(lambda x: 10.0 ** x)


class Normalizer:
    scaler = MinMaxScaler()

    def __init__(self, train_data):
        self.scaler.fit(train_data)

    def get_normalized_data(self, dataset):
        return self.scaler.transform(dataset)


import pandas as pd
import numpy as np
np.set_printoptions(suppress=True)

from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score

from xgboost import XGBRegressor, DMatrix, cv
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data = pd.read_csv("./data/train.csv")
    test_data = pd.read_csv("./data/test.csv")

    train_data = data.iloc[:, :-1]
    train_label = data.iloc[:, -1]

    train_num = train_data.shape[0]
    all_data = pd.concat([train_data, test_data])
    all_data = process_data(all_data)

    train_data = all_data.iloc[:train_num, :]
    test_data = all_data.iloc[train_num:, :]

    train_data.to_csv("train_processed.csv")
    test_data.to_csv("test_processed.csv")

    # normalization
    # feature_normalizer = Normalizer(train_data)
    # train_data = pd.DataFrame(feature_normalizer.get_normalized_data(train_data), columns=all_data.columns)
    # test_data = pd.DataFrame(feature_normalizer.get_normalized_data(test_data), columns=all_data.columns)

    train_data.to_csv("train_processed_normal.csv")
    test_data.to_csv("test_processed_normal.csv")

    train_label.hist()


    # regressor
    all_train_set = pd.concat([train_data, train_label], axis=1, sort=False)
    # print(all_train_set.describe())
    # corr between different attributes
    corr_matrix = all_train_set.corr()
    ax = plt.matshow(corr_matrix)
    plt.colorbar(ax)
    plt.show()
    # print(all_train_set.columns)

    # print(train_set.shape, train_label.shape, test_set.shape, test_label.shape)
    # print(train_set, train_label)

    X = all_train_set.iloc[:, :-1]
    Y = all_train_set.iloc[:, -1]
    print(X.shape, Y.shape)

    xgb_params = {
        # 'min_child_weight': 1,
        'eta': 0.001,  # < 0.6
        # 'colsample_bytree': 0.6,
        'max_depth': 10,
        'subsample': 0.75,
        'lambda': 0.9,  # regularization
        #        'alpha': 12, # regularization
        'gamma': 3.5,  # Gamma specifies the minimum loss reduction required to make a split.
        'silent': 1,
        'verbose_eval': True,
        #         'seed': 18,
        'nthread': 10,
        'eval_metric': 'rmse',
                'scale_pos_weight': 1,
        #        'max_delta_step': 1,
        'n_jobs': 10,
    }
    max_rounds = 12000
    early_stop_rounds = 20

    regressor = XGBRegressor(**xgb_params)
    xgtrain = DMatrix(X, label=Y)
    xgresult = cv(xgb_params, xgtrain, nfold=20, verbose_eval=50, num_boost_round=max_rounds,
                  early_stopping_rounds=early_stop_rounds)
    regressor.set_params(n_estimators=xgresult.shape[0])

    # regressor = SVR(degree=8, C=1, epsilon=0.2)
    regressor.fit(X, Y)
    scores = cross_val_score(regressor, X, Y, cv=20, scoring='neg_mean_squared_error')
    print("feature importances:")
    print(pd.Series(regressor.feature_importances_, index=all_data.columns).sort_values(ascending=False))

    train_predict_label = pd.DataFrame(regressor.predict(X))
    print("train MSE:", mean_squared_error(regressor.predict(X), Y))
    print("cv MSE:", scores, np.mean(scores))

    # train_predict_label_real = train_predict_label.apply(lambda x: resume_from_log_label(x))
    # print("real train MSE", mean_squared_error(train_predict_label_real, Y))

    predict_result = regressor.predict(test_data)
    predict_result[predict_result<0] = train_label.min()
    # pd.DataFrame(predict_result, columns=['time']).to_csv("result.csv")
    # print(predict_result)

    predict_result = pd.DataFrame(predict_result, columns=['time'])
    # predict_result = predict_result.apply(lambda x: resume_from_log_label(x))
    predict_result.to_csv("result_%2.2f.csv" % (np.mean(scores)))
    pd.concat([test_data, predict_result], axis=1).to_csv('result_full.csv')
    print(np.array(predict_result))
