from autogoal.datasets import haha
from autogoal.kb import Seq, Sentence, Supervised, VectorCategorical
from autogoal.ml import AutoML
from autogoal.utils import Gb, Min, Hour
from autogoal.search import NSPESearch
from metrics import f_score_vs_train_time
from utils import plot_and_save, print_and_save

automl = AutoML(
    input=(Seq[Sentence], Supervised[VectorCategorical]),
    output=VectorCategorical,
    pop_size=40,
    score_metric=f_score_vs_train_time("binary"),
    search_algorithm=NSPESearch,
    maximize=(True, True),
    random_state=2,
    number_of_solutions=100,
    memory_limit=16 * Gb,
    search_timeout=8 * Hour,
)


xtrain, ytrain, xtest, ytest = haha.load()
automl.fit(xtrain, ytrain)

print_and_save(automl, "haha_fscore_vs_time")
plot_and_save(
    automl.best_score_,
    automl.solutions_fns_trace,
    xlabel="f1 score",
    ylabel="train time",
    folder_name="haha_fscore_vs_time",
)
