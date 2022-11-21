from autogoal.datasets import cars
from autogoal.kb import MatrixContinuousDense, Supervised, VectorCategorical
from autogoal.ml import AutoML
from autogoal.utils import Gb, Min
from autogoal.search import NSPESearch
from metrics import acc_vs_train_time
from utils import plot_and_save, print_and_save

automl = AutoML(
    input=(MatrixContinuousDense, Supervised[VectorCategorical]),
    output=VectorCategorical,
    pop_size=40,
    score_metric=acc_vs_train_time,
    search_algorithm=NSPESearch,
    maximize=(True, True),
    random_state=2,
    number_of_solutions=100,
    memory_limit=16 * Gb,
    search_timeout=30 * Min,
)

x, y = cars.load()
automl.fit(x, y)

print_and_save(automl)
plot_and_save(
    automl.best_score_,
    automl.solutions_fns_trace,
    xlabel="accuracy",
    ylabel="training time",
    path="./graphics/cars_acc_vs_time.png",
)
