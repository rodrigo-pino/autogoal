from autogoal.datasets import meddocan
from autogoal.kb import Supervised, VectorCategorical, Seq, Word, Label
from autogoal.ml import AutoML
from autogoal.utils import Gb, Hour, Min
from autogoal.search import NSPESearch
from metrics import meddocan_fscore_vs_train_time
from utils import plot_and_save, print_and_save

automl = AutoML(
    input=(Seq[Seq[Word]], Supervised[Seq[Seq[Label]]]),
    output=Seq[Seq[Label]],
    pop_size=40,
    score_metric=meddocan_fscore_vs_train_time,
    search_algorithm=NSPESearch,
    maximize=(True, True),
    random_state=2,
    number_of_solutions=100,
    memory_limit=16 * Gb,
    search_timeout=10 * Hour,
    evaluation_timeout=10 * Min,
)

xtrain, ytrain, xvalid, yvalid = meddocan.load()
automl.fit(xtrain, ytrain)

print_and_save(automl, "meddocan_fscore_vs_train")
plot_and_save(
    automl.best_score_,
    automl.solutions_fns_trace,
    xlabel="fscore",
    ylabel="train time",
    folder_name="meddocan_fscore_vs_train",
)
