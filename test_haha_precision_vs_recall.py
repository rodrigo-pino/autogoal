from autogoal.datasets import haha
from autogoal.kb import Seq, Sentence, Supervised, VectorCategorical
from autogoal.ml import AutoML
from autogoal.utils import Gb, Hour, Min
from autogoal.search import NSPESearch
from metrics import precision_vs_recall
from utils import plot_and_save, print_and_save

automl = AutoML(
    input=(Seq[Sentence], Supervised[VectorCategorical]),
    output=VectorCategorical,
    pop_size=40,
    score_metric=precision_vs_recall("binary"),
    search_algorithm=NSPESearch,
    maximize=(True, True),
    random_state=2,
    number_of_solutions=100,
    memory_limit=16 * Gb,
    search_timeout=10 * Hour,
    evaluation_timeout=3 * Min,
)


xtrain, ytrain, xtest, ytest = haha.load()
automl.fit(xtrain, ytrain)


print_and_save(automl, "haha_precision_vs_recall")
plot_and_save(
    automl.best_score_,
    automl.solutions_fns_trace,
    xlabel="precision",
    ylabel="recall",
    folder_name="haha_precision_vs_recall",
)
