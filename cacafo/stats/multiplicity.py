"""
This file contains types and utilities for working with stratified
samples of sources with multiplicity.

The population of interest are 'individuals'; but these individuals
are sampled *through* 'sources'. Each source can report information
about multiple individuals, and each individual can be reported by
multiple sources.

The sources are divided into sample strata, but the individuals may
be reported by sources in different strata.

For more information, see:

Sirken, Monroe G. “Stratified Stratum Surveys with Multiplicity.”
Journal of the American Statistical Association 67, no. 337 (1972): 224–27.
https://doi.org/10.2307/2284732.
"""

from collections import namedtuple
from dataclasses import dataclass
from typing import ClassVar, Dict, List, Optional, Set, Tuple

import numpy as np


@dataclass
class Individual:
    """
    An individual is a unit of interest that can be reported by sources.

    Attributes:
        individual_id: The identifier for the individual.
        num_reported_by: The number of sources that report the individual.
            The number *needs to be complete, including sampled and unsampled sources*.
    """

    individual_id: int
    num_reported_by: int = 0

    _individuals: ClassVar[Dict[int, "Individual"]] = {}

    @staticmethod
    def max_id():
        if not Individual._individuals:
            return 0
        return max(Individual._individuals.keys())

    def __init__(self, num_reported_by: int):
        self.individual_id = Individual.max_id() + 1
        self.num_reported_by = num_reported_by
        Individual._individuals[self.individual_id] = self

    def __hash__(self):
        return hash(self.individual_id)


@dataclass
class Source:
    """
    A source is a sample unit that reports information about individuals.

    Attributes:
        individuals: A list of individuals reported to be of interest by the source.
        stratum: The stratum identifier for the source.
    """

    reported_individuals: Set[Individual]

    def weighted_num_reported(self):
        return sum(
            [1 / individual.num_reported_by for individual in self.reported_individuals]
        )


@dataclass
class Stratum:
    """
    A stratum is a collection of sources that are sampled together.

    Attributes:
        sampled_sources: A list of sources.
        total_number_of_sources_in_stratum: The total number of sources in the stratum
    """

    sampled_sources: List[Source]
    unsampled_labeled_sources: List[Source]
    total_number_of_sources_in_stratum: int

    def biased_estimate(self):
        return (
            self.total_number_of_sources_in_stratum
            / len(self.sampled_sources)
            * sum([source.weighted_num_reported() for source in self.sampled_sources])
        )

    def stochastic_biased_estimate(self, seed=None):
        if seed:
            np.random.seed(seed)
        prevalence = len(
            [source for source in self.sampled_sources if source.reported_individuals]
        ) / len(self.sampled_sources)
        stochastic_num_positives = np.random.binomial(
            len(self.sampled_sources), prevalence
        )
        all_positives = [
            source for source in self.sampled_sources if source.reported_individuals
        ] + [
            source
            for source in self.unsampled_labeled_sources
            if source.reported_individuals
        ]
        stochastic_positives = np.random.choice(
            all_positives, size=stochastic_num_positives, replace=True
        )
        return (
            self.total_number_of_sources_in_stratum
            / len(self.sampled_sources)
            * sum([source.weighted_num_reported() for source in stochastic_positives])
        )


EstimateWithConfidence = namedtuple(
    "EstimateWithConfidence", ["estimate", "lower_bound", "upper_bound"]
)


@dataclass
class Survey:
    """
    A survey is a collection of strata.

    Attributes:
        strata: A dictionary of strata keyed by stratum identifier.
    """

    strata: List[Stratum]

    def unbiased_population_estimate(
        self, bootstrap_iters: Optional[int] = 1000
    ) -> EstimateWithConfidence:
        point = sum([stratum.biased_estimate() for stratum in self.strata])
        estimates = []
        for _ in range(bootstrap_iters):
            estimates.append(
                sum([stratum.stochastic_biased_estimate() for stratum in self.strata])
            )
        estimates = np.array(estimates)
        estimate = EstimateWithConfidence(
            estimate=point,
            lower_bound=np.percentile(estimates, 2.5),
            upper_bound=np.percentile(estimates, 97.5),
        )
        return estimate


TestResult = namedtuple("TestResult", ["estimate", "true_population_size"])


def random_test(true_population_size=None, bootstrap_iters=1000) -> TestResult:
    if not true_population_size:
        true_population_size = np.random.randint(100, 2000)
    individuals = [Individual(0) for _ in range(true_population_size)]
    sources = [
        Source(
            [
                individuals[i]
                for i in np.random.choice(
                    len(individuals), size=np.random.randint(1, 10), replace=False
                )
            ]
        )
        for _ in range(np.random.randint(100, 300))
    ]
    for i in individuals:
        i.num_reported_by = sum(
            [i in source.reported_individuals for source in sources]
        )
        if i.num_reported_by == 0:
            sources[np.random.randint(0, len(sources))].reported_individuals.append(i)
            i.num_reported_by = 1
    strata = []
    original_sources_len = len(sources)
    while sources:
        stratum_size = min(
            np.random.randint(1, original_sources_len // 2), len(sources)
        )
        stratum_sources = np.random.choice(sources, size=stratum_size, replace=False)
        strata.append(
            Stratum(
                sampled_sources=stratum_sources,
                total_number_of_sources_in_stratum=len(stratum_sources),
                unsampled_labeled_sources=[],
            )
        )
        sources = [source for source in sources if source not in stratum_sources]
    for s in strata:
        s.sampled_sources = np.random.choice(
            s.sampled_sources,
            size=np.random.randint(1, len(s.sampled_sources) // 2 + 2),
            replace=False,
        )

    survey = Survey(strata)
    return TestResult(
        estimate=survey.unbiased_population_estimate(bootstrap_iters=bootstrap_iters),
        true_population_size=len(individuals),
    )


def histogram_test(num_tests: int = 500) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    import tqdm

    estimates = []
    for _ in tqdm.tqdm(range(num_tests)):
        test_result = random_test(1000, 1)
        estimates.append(test_result.estimate.estimate)
    sns.histplot(estimates, kde=True)
    plt.vlines(test_result.true_population_size, 0, 0.01, colors="r")
    plt.show()
