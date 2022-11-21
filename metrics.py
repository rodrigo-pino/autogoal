from typing import Tuple
from time import time
import math
import sklearn.metrics as sk_metrics
import autogoal.ml.metrics as metrics

# Metric 1
def acc_vs_train_time(*args, **kwargs):
    t1 = time()
    acc = metrics.accuracy(*args, **kwargs)
    t2 = time()
    score = [acc, t1 - t2]
    return score


# Metric 2
def accurayc_vs_not_accuracy(*args, **kwargs):
    acc = metrics.accuracy(*args, **kwargs)
    return [acc, 1 - acc] if acc > 0 else [-math.inf, -math.inf]


# Metric 3
def f_score_vs_train_time(average: str):
    @metrics.supervised_fitness_fn
    def fitness_f_score(ytrue, ypred) -> float:
        try:
            r = sk_metrics.f1_score(ytrue, ypred, average=average)
            return r
        except Exception as e:
            raise e

    def f_score_vs_train_time_inner(*args, **kwargs):
        t1 = time()
        try:
            f = fitness_f_score(*args, **kwargs)
        except Exception as e:
            # print(e)
            raise e
        t2 = time()
        score = [f, t1 - t2]
        return score

    return f_score_vs_train_time_inner


# Metric 4
def precision_vs_recall(average: str):
    @metrics.supervised_fitness_fn
    def precision_vs_recall_inner(ytrue, ypred) -> Tuple[float, float]:
        precision = sk_metrics.precision_score(ytrue, ypred, average=average)
        recall = sk_metrics.recall_score(ytrue, ypred, average=average)
        return precision, recall

    return precision_vs_recall_inner
