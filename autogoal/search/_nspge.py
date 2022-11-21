import math
from typing import List, Optional

from autogoal.utils import Gb, Min, Sec
from ._pge import PESearch


# TODO: scale function results in crowding distance
# TODO: Return multiple possible pipelines, instead of just the one
# TODO: How is the genotype updated, since we are not using the best, but a list of the bests.
#       Does it gets updated *N* times with the *N* fittest??
# TODO: Where/When is the population cropped


class NSPESearch(PESearch):
    def __init__(
        self,
        generator_fn=None,
        fitness_fn=None,
        pop_size=20,
        maximize=True,
        errors="raise",
        early_stop=0.5,
        evaluation_timeout: int = 10 * Sec,
        memory_limit: int = 4 * Gb,
        search_timeout: int = 5 * Min,
        target_fn=None,
        allow_duplicates=True,
        number_of_solutions=None,
        ranking_fn=None,
        learning_factor=0.05,
        selection: float = 0.2,
        epsilon_greed: float = 0.1,
        random_state: Optional[int] = None,
        name: str = None,
        save: bool = False,
        **kwargs,
    ):
        def default_ranking_fn(_, fns):
            rankings = [-math.inf] * len(fns)
            fronts = self.non_dominated_sort(fns)
            # return fronts[0]
            for ranking, front in enumerate(fronts):
                for index in front:
                    rankings[index] = -ranking
            return rankings

        if ranking_fn is None:
            ranking_fn = default_ranking_fn

        super().__init__(
            generator_fn=generator_fn,
            fitness_fn=fitness_fn,
            pop_size=pop_size,
            maximize=maximize,
            errors=errors,
            early_stop=early_stop,
            evaluation_timeout=evaluation_timeout,
            memory_limit=memory_limit,
            search_timeout=search_timeout,
            target_fn=target_fn,
            allow_duplicates=allow_duplicates,
            number_of_solutions=number_of_solutions,
            ranking_fn=ranking_fn,
            learning_factor=learning_factor,
            selection=selection,
            epsilon_greed=epsilon_greed,
            random_state=random_state,
            name=name,
            save=save,
            **kwargs,
        )

    def _indices_of_fittest(self, fns: List[List[float]]):
        print("Indices of  fittest")
        fronts = self.non_dominated_sort(fns)
        indices = []
        k = int(self._selection * len(fns))

        for front in fronts:
            if len(indices) + len(front) <= k:
                indices.extend(front)
            else:
                indices.extend(
                    sorted(front, key=lambda i: -self.crowding_distance(fns, front, i))[
                        : k - len(indices)
                    ]
                )
                break
        return indices

    def non_dominated_sort(self, scores: List[List[float]]):
        fronts: List[List[int]] = [[]]
        domination_rank = [0] * len(scores)
        dominated_scores = [list() for _ in scores]

        for i, score_i in enumerate(scores):
            for j, score_j in enumerate(scores):
                if self._improves(score_i, score_j):
                    dominated_scores[i].append(j)
                elif self._improves(score_j, score_i):
                    domination_rank[i] += 1
            if domination_rank[i] == 0:
                fronts[0].append(i)

        front_rank = 0
        while len(fronts[front_rank]) > 0:
            next_front = []
            for i in fronts[front_rank]:
                for dominated in dominated_scores[i]:
                    domination_rank[dominated] -= 1
                    if domination_rank[dominated] == 0:
                        next_front.append(dominated)
            front_rank += 1
            fronts.append(next_front)

        return fronts[:-1]

    def crowding_distance(
        self, scores: List[List[float]], front: List[int], index: int
    ) -> float:
        if len(front) <= 0:
            raise ValueError("Pareto front is empty or negative")
        if isinstance(self._maximize, bool):
            self._maximize = (self._maximize,)

        scaled_scores = feature_scaling(scores)

        crowding_distances: List[float] = [0 for _ in scores]
        for m in range(len(self._maximize)):
            front = sorted(front, key=lambda x: scores[x][m])
            crowding_distances[front[0]] = math.inf
            crowding_distances[front[-1]] = math.inf
            m_values = [scaled_scores[i][m] for i in front]
            scale: float = max(m_values) - min(m_values)
            if scale == 0:
                scale = 1
            for i in range(1, len(front) - 1):
                crowding_distances[i] += (
                    scaled_scores[front[i + 1]][m] - scaled_scores[front[i - 1]][m]
                ) / scale

        return crowding_distances[index]


def feature_scaling(solutions_scores: List[List[float]]) -> List[List[float]]:
    total_metrics = len(solutions_scores[0])
    scaled_scores = [list() for _ in solutions_scores]

    metric_selector = 0
    while metric_selector < total_metrics:
        # All scores per solution
        # sol1: [1, 2]
        # sol2: [3, 4]
        # m_score[0] -> [1, 3]
        # m_score[1] -> [3, 4]
        m_scores = [score[metric_selector] for score in solutions_scores]
        if len(m_scores) == 1:
            for scaled in scaled_scores:
                scaled.append(1)
            metric_selector += 1
            continue

        # print("----- metric:", metric_selector, [x for x in m_scores if x != -math.inf])
        filtered_m_scores = [v for v in m_scores if v != -math.inf]
        if len(filtered_m_scores) == 0:
            for scaled in scaled_scores:
                scaled.append(-math.inf)
            metric_selector += 1
            continue

        max_value = max(filtered_m_scores)
        min_value = min(filtered_m_scores)
        diff = max_value - min_value

        # When there is just one valid solution (everyone else is minus infinity)
        if diff == 0:
            index = m_scores.index(max_value)
            for i, scaled in enumerate(scaled_scores):
                if i == index or m_scores[i] != -math.inf:
                    scaled.append(1)
                else:
                    scaled.append(-math.inf)
            metric_selector += 1
            continue

        for i, scaled in enumerate(scaled_scores):
            scaled_value = (
                m_scores[i] - min_value
            ) / diff  # if m_scores[i] != -math.inf else 0
            scaled.append(scaled_value)
        metric_selector += 1

    return scaled_scores
