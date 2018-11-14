# encoding:utf-8
# 生成策略：在正常随机采样生成的同时，增加几种采样：加大N_SAMPLE的下界,加大N_FEATURE的下界，加大MAX_ITER的下界
# 这是为了生成更多运行时间长的训练数据，因为这些数据的准确度倾向于更影响整体的表现。
##

from sklearn.datasets import make_classification
from sklearn.linear_model import SGDClassifier
from scipy.stats import randint as sp_randint
from scipy.stats import rv_discrete
import numpy as np
import pandas as pd
import time
import random
import itertools
from copy import deepcopy
from math import log2

# features = sp_randint(10, 30)

df = pd.DataFrame(columns=['penalty', 'l1_ratio', 'alpha', 'max_iter', 'random_state', 'n_jobs', 'n_samples',
                           'n_features', 'n_classes', 'n_clusters_per_class', 'n_informative', 'flip_y', 'scale',
                           'time'])

rows = []
penalty_items = ['none', 'l1', 'elasticnet', 'l2']
alpha_range = [0.0001, 0.001, 0.01]
jobs_range = [1, 2, 4]



def generate_runtime_simple(n_samples, n_features, n_classes, n_clusters_per_class,
                 penalty, alpha, l1_ratio, max_iter, n_jobs):
    x, y = make_classification(n_samples=n_samples, n_features=n_features,
                               n_classes=n_classes,
                               n_clusters_per_class=n_clusters_per_class,
                               n_informative=int(log2(n_classes * n_clusters_per_class)+1))
    model = SGDClassifier(
        penalty=penalty, alpha=alpha, l1_ratio=l1_ratio,
        max_iter=max_iter,
        n_jobs=n_jobs)
    tick1 = time.time()
    model.fit(x, y)
    tick2 = time.time()
    time_interval = tick2 - tick1
    return time_interval


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
        # n_informative = row.n_informative
        n_classes = row.n_classes
        n_clusters_per_class = row.n_clusters_per_class
        # flip_y = row.flip_y
        # scale = row.scale
        penalty = row.penalty
        alpha = row.alpha
        l1_ratio = row.l1_ratio
        max_iter = row.max_iter
        n_jobs = row.n_jobs
        # random_state = row.random_state
        train_data.loc[i, "newtime"] = generate_runtime_simple(n_samples, n_features, n_classes,
                                                        n_clusters_per_class, penalty, alpha,
                                                        l1_ratio, max_iter, n_jobs)
        cnt += 1
        print("generate %d data" % cnt)
    return train_data


def run_train_augment(filename):
    train_data = pd.read_csv(filename)
    new_train_data = get_local_data(train_data)
    new_train_data.to_csv("new_train_data.csv")


# choose value from to_choose to augment chosen, and finally add it to value_list
def choose_value(chosen, to_choose, value_list):
    if len(to_choose) == 0:
        value_list.append(chosen)
    else:
        k = list(to_choose.keys())[0]
        for v in to_choose[k]:
            new_chosen = deepcopy(chosen)
            new_to_choose = deepcopy(to_choose)
            new_to_choose.pop(k)
            new_chosen[k] = v
            choose_value(new_chosen, new_to_choose, value_list)
    return


def run_generate_big_train(filename):
    train_data = pd.read_csv(filename)
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    newdf = train_data.select_dtypes(include=numerics)
    #for k in newdf:
        # train_data[k].plot.hist(bins=10, rwidth=0.8)
        # plt.title("freq on %s" % k)
        # plt.show()
    # 发现除了n_informative是正态分布，其他都是均匀分布
    # 有几个属性生成的时候使用全部的unique值，因为比较少
    # 假设n_informative没什么用，生成的时候不考虑，那么就其他的来说，直接最小值最大值之间生成就可以了
    skip_attr = ['n_informative', 'random_state', 'id', 'scale', 'flip_y', 'time']
    uni_attr = []
    sample_attr = []
    new_values = {}
    sample_num = 2
    total_scale = 1
    for k in train_data.columns:
        if k in skip_attr:
            continue
        print(total_scale)

        print(k)
        if len(list(train_data[k].unique())) < 100:
            uni_attr.append(k)
            print(len(list(train_data[k].unique())))
            new_values[k] = list(train_data[k].unique())
            total_scale *= len(list(train_data[k].unique()))
            continue
        else:
            sample_attr.append(k)
            min_value = train_data[k].min()
            max_value = train_data[k].max()
            step = (max_value - min_value) / sample_num
            print(min_value, max_value, step)
            new_values[k] = []
            for v in np.arange(min_value, max_value, step):
                new_values[k].append(v if k in ['l1_ratio'] else int(v))
            total_scale *= sample_num
    print(total_scale)
    new_data = []
    choose_value({}, new_values, new_data)
    new_data = pd.DataFrame(new_data)
    new_data.to_csv("new_middle_data.csv")
    # print(new_data)


def generate_random_sample(mode):
    maxIter = 10
    for i in range(maxIter):
        print("mode:", mode, "iter:", i)
        if mode == 1:
            samples = sp_randint(700, 1500)
        else:
            samples = sp_randint(100, 1500)
        n_samples = samples.rvs()

        if mode == 2:
            n_features = random.randint(700, 1500)  # 100-1500
        else:
            n_features = random.randint(100, 1500)

        n_penalty = random.randint(0, 3)
        penalty = penalty_items[n_penalty]
        l1_ratio = random.random()
        n_alpha = random.randint(0, 2)
        alpha = alpha_range[n_alpha]
        if mode == 3:
            max_iter = random.randint(700, 1000)  # 100-1000
        else:
            max_iter = random.randint(100, 1000)

        random_state = random.randint(0, 1000)
        n_job = random.randint(0, len(jobs_range) - 1)
        jobs = jobs_range[n_job]

        n_classes = random.randint(2, 12)

        n_clusters_per_class = random.randint(2, 6)
        n_informative = random.randint(5, 12)
        flip_y = random.random() / 10.0
        scale = random.uniform(1, 100)

        if n_classes * n_clusters_per_class > 2 ** n_informative:
            continue

        x, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_informative,
                                   n_classes=n_classes,
                                   n_clusters_per_class=n_clusters_per_class,
                                   flip_y=flip_y, class_sep=1.0,
                                   scale=scale)
        model = SGDClassifier(
            penalty=penalty, alpha=alpha, l1_ratio=l1_ratio,
            max_iter=max_iter,
            n_jobs=jobs,
            random_state=random_state)
        tick1 = time.time()
        model.fit(x, y)
        tick2 = time.time()
        time_interval = tick2 - tick1

        rows.append(
            [penalty, l1_ratio, alpha, max_iter, random_state, jobs, n_samples, n_features, n_classes, n_clusters_per_class,
             n_informative, flip_y, scale, time_interval])

    for row in rows:
        df.loc[len(df)] = row

    df.to_csv('time_mode%d.csv' % mode)


if __name__ == "__main__":
    # run_generate_big_train("train.csv")
    # run_train_augment("new_middle_data.csv")
    generate_random_sample(0)
