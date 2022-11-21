from typing import List
from autogoal.ml import AutoML
from pathlib import Path
import matplotlib.pyplot as plt
import os
import shutil


# plot all points in trace
# points in best_score are marked as part of the Pareto Front
# path inidicates where to store the image
def plot_and_save(
    best_scores,
    trace,
    xlabel="metric 0",
    ylabel="metric 1",
    path="./graphics/generic.jpg",
):
    print("------------- Plotting -------------------")
    total_solutions = len(trace)
    stages = int(total_solutions / 10)
    stages = 1 if stages == 0 else stages

    ax = plt.subplot()
    for i in range(0, total_solutions):
        solutions_scores = trace[i]
        xs, ys = restructure(solutions_scores)
        (sols,) = ax.plot(xs, ys, "o", color=str(1 - i / total_solutions))
        if i == int(total_solutions / 2):
            sols.set_label("Solutions")

    # Mark solutions that belongs to the Pareto front
    # plt.colorbar([0, 0.5, 1])
    px, py = restructure(best_scores)
    ax.plot(px, py, "x", color="red", label="Pareto Front")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best")
    plt.savefig(path, dpi=1200)
    plt.show()


# Shows best solutions obtained by AutoGOAL
def print_and_save(automl: AutoML, folder_name="generic"):
    print(f"-------------- Solutions ({len(automl.best_pipeline_)}) -----------------")

    path = Path(f"./pipelines/{folder_name}")
    create_empty_folder(path)

    count = 0
    for pipeline, score in zip(automl.best_pipeline_, automl.best_score_):
        count += 1
        # Printing
        print("\nSolution:", count)
        print(score)
        print(pipeline)

        # Saving
        arg_path = path / f"solution_{count}"
        create_empty_folder(arg_path)
        pipeline.save_algorithms(arg_path)


def create_empty_folder(path: Path):
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass
    os.mkdir(path)


# Example input: [[1, 2], [3, 4]]
# Example output: [[1, 3], [2, 4]]
def restructure(solutions_scores: List[List[float]]):
    l1 = []
    l2 = []
    for [x, y] in solutions_scores:
        l1.append(x)
        l2.append(y)
    return l1, l2
