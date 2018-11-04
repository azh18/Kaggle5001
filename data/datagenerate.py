from sklearn.datasets import make_classification
from sklearn.linear_model import SGDClassifier
from scipy.stats import randint as sp_randint
from scipy.stats import rv_discrete
import numpy as np
import pandas as pd
import time
import random

# features = sp_randint(10, 30)

df = pd.DataFrame(columns=['penalty', 'l1_ratio', 'alpha', 'max_iter', 'random_state', 'n_jobs', 'n_samples',
                           'n_features', 'n_classes', 'n_clusters_per_class', 'n_informative', 'flip_y', 'scale',
                           'time'])

rows = []
penalty_items = ['none', 'l1', 'elasticnet', 'l2']
alpha_range = [0.0001, 0.001, 0.01]
jobs_range = [1, 2, 4]

samples = sp_randint(100, 300)


def generate_runtime(n_samples, n_features, n_informative, n_classes, n_clusters_per_class,
                 flip_y, scale, penalty, alpha, l1_ratio, max_iter, n_jobs, random_state):
    x, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_informative,
                               n_classes=n_classes,
                               n_clusters_per_class=n_clusters_per_class,
                               flip_y=flip_y, class_sep=1.0,
                               scale=scale)
    model = SGDClassifier(
        penalty=penalty, alpha=alpha, l1_ratio=l1_ratio,
        max_iter=max_iter,
        n_jobs=n_jobs,
        random_state=random_state)
    tick1 = time.time()
    model.fit(x, y)
    tick2 = time.time()
    time_interval = tick2 - tick1
    return time_interval


def generate_row(n_samples, n_features, n_informative, n_classes, n_clusters_per_class,
                 flip_y, scale, penalty, alpha, l1_ratio, max_iter, n_jobs, random_state):
    x, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_informative,
                               n_classes=n_classes,
                               n_clusters_per_class=n_clusters_per_class,
                               flip_y=flip_y, class_sep=1.0,
                               scale=scale)
    model = SGDClassifier(
        penalty=penalty, alpha=alpha, l1_ratio=l1_ratio,
        max_iter=max_iter,
        n_jobs=n_jobs,
        random_state=random_state)
    tick1 = time.time()
    model.fit(x, y)
    tick2 = time.time()
    time_interval = tick2 - tick1
    return [penalty, l1_ratio, alpha, max_iter, random_state, n_jobs, n_samples, n_features, n_classes,
            n_clusters_per_class, n_informative, flip_y, scale, time_interval]


def get_local_data(train_data):
    cnt = 0
    for i in train_data.index:
        row = train_data.loc[i, :]
        n_samples = row.n_samples
        n_features = row.n_features
        n_informative = row.n_informative
        n_classes = row.n_classes
        n_clusters_per_class = row.n_clusters_per_class
        flip_y = row.flip_y
        scale = row.scale
        penalty = row.penalty
        alpha = row.alpha
        l1_ratio = row.l1_ratio
        max_iter = row.max_iter
        n_jobs = row.n_jobs
        random_state = row.random_state
        train_data.loc[i, "newtime"] = generate_runtime(n_samples, n_features, n_informative, n_classes,
                                                        n_clusters_per_class, flip_y, scale, penalty, alpha,
                                                        l1_ratio, max_iter, n_jobs, random_state)
        cnt += 1
        print("generate %d data" % cnt)
    return train_data


def run(filename):
    train_data = pd.read_csv(filename)
    new_train_data = get_local_data(train_data)
    new_train_data.to_csv("new_train_data.csv")


if __name__ == "__main__":
    run("train.csv")



#
# maxIter = 10
# for i in range(maxIter):
#     n_samples = samples.rvs()
#     n_features = random.randint(100, 1500)
#     n_penalty = random.randint(0, 3)
#     penalty = penalty_items[n_penalty]
#     l1_ratio = random.random()
#     n_alpha = random.randint(0, 2)
#     alpha = alpha_range[n_alpha]
#     max_iter = random.randint(100, 1000)
#     random_state = random.randint(0, 1000)
#     n_job = random.randint(0, len(jobs_range) - 1)
#     jobs = jobs_range[n_job]
#
#     n_classes = random.randint(2, 12)
#
#     n_clusters_per_class = random.randint(2, 6)
#     n_informative = random.randint(5, 12)
#     flip_y = random.random() / 10.0
#     scale = random.uniform(1, 100)
#
#     x, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_informative,
#                                n_classes=n_classes,
#                                n_clusters_per_class=n_clusters_per_class,
#                                flip_y=flip_y, class_sep=1.0,
#                                scale=scale)
#     model = SGDClassifier(
#         penalty=penalty, alpha=alpha, l1_ratio=l1_ratio,
#         max_iter=max_iter,
#         n_jobs=jobs,
#         random_state=random_state)
#     tick1 = time.time()
#     model.fit(x, y)
#     tick2 = time.time()
#     time_interval = tick2 - tick1
#
#     rows.append(
#         [penalty, l1_ratio, alpha, max_iter, random_state, jobs, n_samples, n_features, n_classes, n_clusters_per_class,
#          n_informative, flip_y, scale, time_interval])

for row in rows:
    df.loc[len(df)] = row

df.to_csv('time.csv')
