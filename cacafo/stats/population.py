from dataclasses import dataclass
from enum import Enum
from typing import Union

import numpy as np
import pandas as pd
import sqlalchemy as sa
from scipy.stats import f
from statsmodels.stats.proportion import proportion_confint

import cacafo.db.sa_models as m
import cacafo.query
from cacafo.db.session import get_sqlalchemy_session

BOOTSTRAP_ITERATIONS = None


def bootstrap_recall(strata_df):
    assert BOOTSTRAP_ITERATIONS, "Number of bootstrap iterations not set"
    distributions = []
    for i, row in strata_df.iterrows():
        if row["stratum"] == "post hoc":
            continue
        n_images = row["n_images"]
        n_positive = row["positive"]
        distributions.append(
            np.random.binomial(
                n_images, n_positive / n_images, size=BOOTSTRAP_ITERATIONS
            )
        )
    distributions = np.array(distributions)
    totals = np.sum(distributions, axis=0)
    return {
        "population": np.median(totals),
        "lower": np.percentile(totals, 2.5),
        "upper": np.percentile(totals, 97.5),
    }


def db_strata_counts():
    session = get_sqlalchemy_session()

    all_image_id_strata = session.execute(
        sa.select(
            m.Image.__table__.c.id,
            m.County.__table__.c.name,
            m.Image.__table__.c.bucket,
        )
        .select_from(m.Image)
        .join(m.County)
        .filter(m.Image.bucket.is_not(None))
    ).all()

    initially_labeled_image_ids = set(
        session.scalars(cacafo.query.initially_labeled_images()).all()
    )
    labeled_image_subquery = cacafo.query.labeled_images().subquery()
    labeled_image_ids = set(
        session.scalars(
            sa.select(labeled_image_subquery.c.id).select_from(labeled_image_subquery)
        ).all()
    )
    unlabeled_image_subquery = cacafo.query.unlabeled_images().subquery()
    unlabeled_image_ids = set(
        session.scalars(
            sa.select(unlabeled_image_subquery.c.id).select_from(
                unlabeled_image_subquery
            )
        ).all()
    )
    positive_image_subquery = cacafo.query.positive_images().subquery()
    positive_image_ids = set(
        session.scalars(
            sa.select(positive_image_subquery.c.id).select_from(positive_image_subquery)
        ).all()
    )

    assert labeled_image_ids & unlabeled_image_ids == set()
    assert positive_image_ids & unlabeled_image_ids == set()
    assert len(labeled_image_ids | unlabeled_image_ids) == len(all_image_id_strata)

    data = {}
    for image_id, county, bucket in all_image_id_strata:
        strata = f"{county}:{bucket}"
        if image_id in initially_labeled_image_ids:
            strata = "completed"
        if not data.get(strata):
            data[strata] = {
                "stratum": strata,
                "unlabeled": 0,
                "labeled": 0,
                "positive": 0,
            }
        if image_id in labeled_image_ids:
            data[strata]["labeled"] += 1
        else:
            data[strata]["unlabeled"] += 1
        if image_id in positive_image_ids:
            data[strata]["positive"] += 1
    df = pd.DataFrame(data.values())
    return df


@dataclass
class Estimate:
    point: Union[float, int] = 0
    lower: Union[float, int] = 0
    upper: Union[float, int] = 0

    def __str__(self):
        return f"{self.point:.2f} ({self.lower:.3f}-{self.upper:.3f})"

    def __repr__(self):
        return f"Estimate(point={self.point}, lower={self.lower}, upper={self.upper})"

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return Estimate(
                point=self.point + other,
                lower=self.lower + other,
                upper=self.upper + other,
            )
        raise NotImplementedError

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return Estimate(
                point=self.point - other,
                lower=self.lower - other,
                upper=self.upper - other,
            )
        raise NotImplementedError

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Estimate(
                point=self.point * other,
                lower=self.lower * other,
                upper=self.upper * other,
            )
        raise NotImplementedError

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return Estimate(
                point=self.point / other,
                lower=self.lower / other,
                upper=self.upper / other,
            )
        raise NotImplementedError

    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            return Estimate(
                point=float("inf") if self.point == 0 else other / self.point,
                lower=float("inf") if self.lower == 0 else other / self.lower,
                upper=other / self.upper,
            )
        raise NotImplementedError


@dataclass
class Stratum:
    name: str
    unlabeled: int
    labeled: int
    positive: int = 0

    def copy(self):
        return Stratum(
            name=self.name,
            unlabeled=self.unlabeled,
            labeled=self.labeled,
            positive=self.positive,
        )

    @property
    def total(self):
        return self.unlabeled + self.labeled

    @property
    def prevalence(self) -> Estimate:
        return Estimate(
            point=self.positive / self.labeled,
            lower=proportion_confint(
                self.positive, self.labeled, alpha=0.05, method="wilson"
            )[0],
            upper=proportion_confint(
                self.positive, self.labeled, alpha=0.05, method="wilson"
            )[1],
        )

    @property
    def population(self) -> Estimate:
        return self.prevalence * self.unlabeled + self.positive

    @property
    def recall(self) -> Estimate:
        return self.positive / self.population

    def sample_prevalence(self, n=1):
        p = np.random.rand(n)
        outputs = []
        for i in range(n):
            alpha = 1 - abs(2 * p[i] - 1)
            outputs.append(
                proportion_confint(
                    self.positive, self.labeled, alpha=alpha, method="beta"
                )
            )
        return np.array(outputs)

    def sample_population(self, n=1):
        return self.sample_prevalence(n) * self.unlabeled + self.positive

    def label(self, n=1, positives=0):
        self.labeled += n
        self.positive += positives

    def labeled(self, n=1, positives=0):
        new = self.copy()
        new.label(n=n, positives=positives)
        return new

    def to_dict(self):
        return {
            "name": self.name,
            "unlabeled": self.unlabeled,
            "labeled": self.labeled,
            "prevalence": self.prevalence.point,
            "positive": self.positive,
            "total": self.total,
        }

    def __str__(self):
        return f"{self.name}: total={self.total}, positive={self.positive}, prevalence={self.prevalence}, population={self.population}, recall={self.recall}"

    def __repr__(self):
        return f"Stratum(name={self.name}, unlabeled={self.unlabeled}, labeled={self.labeled}, positive={self.positive})"

    def __add__(self, other):
        if isinstance(other, Stratum):
            return Stratum(
                name=f"{self.name}+{other.name}",
                unlabeled=self.unlabeled + other.unlabeled,
                labeled=self.labeled + other.labeled,
                positive=self.positive + other.positive,
            )
        raise NotImplementedError


@dataclass
class Survey:
    strata: list[Stratum]
    post_hoc_positive: int
    bootstrap_iterations: int = 100

    def copy(self):
        return Survey(
            strata=[stratum.copy() for stratum in self.strata],
            post_hoc_positive=self.post_hoc_positive,
            bootstrap_iterations=self.bootstrap_iterations,
        )

    @classmethod
    def from_db(cls):
        df = db_strata_counts()
        strata = []
        post_hoc_positive = 0
        for i, row in df.iterrows():
            if row["stratum"] == "post hoc":
                post_hoc_positive = row["positive"]
                continue
            if not row["stratum"]:
                continue
            stratum = Stratum(
                name=row["stratum"],
                unlabeled=row["unlabeled"],
                labeled=row["labeled"],
                positive=row["positive"],
            )
            strata.append(stratum)
        return cls(strata=strata, post_hoc_positive=post_hoc_positive)

    def to_df(self):
        strata_dicts = [stratum.to_dict() for stratum in self.strata]
        post_hoc = {
            "name": "post hoc",
            "unlabeled": 0,
            "labeled": self.post_hoc_positive,
            "prevalence": 0,
            "positive": self.post_hoc_positive,
            "total": 0,
        }
        strata_dicts.append(post_hoc)
        return pd.DataFrame(strata_dicts)

    def aggregated(self):
        post_hoc = self.post_hoc_positive
        completed_strata = [
            stratum for stratum in self.strata if stratum.name == "completed"
        ]
        lowest_sampling_rate_0 = min(
            stratum.labeled / stratum.total
            for stratum in self.strata
            if "0:" in stratum.name
        )
        lowest_sampling_rate_1 = min(
            stratum.labeled / stratum.total
            for stratum in self.strata
            if "1:" in stratum.name
        )
        super_stratum_0 = Stratum(
            name="super_stratum_0",
            unlabeled=0,
            labeled=0,
            positive=0,
        )
        super_stratum_1 = Stratum(
            name="super_stratum_1",
            unlabeled=0,
            labeled=0,
            positive=0,
        )
        for stratum in self.strata:
            if stratum.unlabeled == 0:
                continue
            lowest_sampling_rate = (
                lowest_sampling_rate_0
                if "0:" in stratum.name
                else lowest_sampling_rate_1
            )
            positive_samples = round(
                lowest_sampling_rate * stratum.total * stratum.prevalence.point
            )
            post_hoc += stratum.positive - positive_samples
            if round(lowest_sampling_rate * stratum.total) == 0:
                raise ValueError(
                    f"Stratum {stratum.name} would have no labeled samples"
                )
            super_stratum = super_stratum_0 if "0:" in stratum.name else super_stratum_1
            super_stratum.labeled += round(lowest_sampling_rate * stratum.total)
            super_stratum.positive += positive_samples
            super_stratum.unlabeled += stratum.total - round(
                lowest_sampling_rate * stratum.total
            )
        new_strata = completed_strata + [super_stratum_0, super_stratum_1]
        return Survey(strata=new_strata, post_hoc_positive=post_hoc)

    def label(self, n=1, positives=0, method: str = "greatest variance"):
        if positives > 0:
            raise NotImplementedError
        match method:
            case "greatest variance":
                strata = sorted(
                    self.strata,
                    key=lambda x: x.population.upper - x.population.lower,
                    reverse=True,
                )
                print("Labeling", strata[0].name)
                strata[0].label(n=n)
            case "least labeled":
                strata = sorted(self.strata, key=lambda x: x.labeled / x.total)
                print("Labeling", strata[0].name)
                strata[0].label(n=n)

    def labeled(self, n=1, positives=0, method: str = "greatest variance"):
        new = self.copy()
        new.label(n=n, positives=positives, method=method)
        return new

    def population(self, method: str = "per_stratum") -> Estimate:
        match method:
            case "per_stratum":
                return stratum_sum_estimator(self)
            case "aggregate":
                return aggregate_estimator(self)
            case "horvitz-thompson":
                return horvitz_thompson_estimator(self)

    def completeness(self, method: str = "per_stratum") -> Estimate:
        strata_positives = np.array([stratum.positive for stratum in self.strata])
        total_positives = np.sum(strata_positives) + self.post_hoc_positive
        est = self.population(method=method)
        return total_positives / est


def stratum_sum_estimator(survey):
    samples = np.array(
        [
            stratum.sample_population(n=survey.bootstrap_iterations)
            for stratum in survey.strata
        ]
    )
    totals = np.sum(samples, axis=0)
    return Estimate(
        point=np.median(totals),
        lower=np.percentile(totals, 2.5),
        upper=np.percentile(totals, 97.5),
    )


def aggregate_estimator(survey):
    return survey.aggregated().population(method="per_stratum")


def horvitz_thompson_estimator(survey):
    raise NotImplementedError
    t = sum(
        (stratum.total / stratum.labeled) * stratum.positive
        for stratum in survey.strata
    )
    return Estimate(t, t, t)


def naip_stratum_f_estimator(survey, alpha=0.05):
    survey_0 = Survey(
        strata=[stratum for stratum in survey.strata if "0:" in stratum.name],
        post_hoc_positive=0,
    )
    survey_1 = Survey(
        strata=[stratum for stratum in survey.strata if "1:" in stratum.name],
        post_hoc_positive=0,
    )
    # completed_survey = Survey(
    #    strata=[stratum for stratum in survey.strata if stratum.unlabeled == 0],
    #    post_hoc_positive=survey.post_hoc_positive,
    # )
    population_0 = stratum_f_estimator(survey_0, alpha=alpha)
    population_1 = stratum_f_estimator(survey_1, alpha=alpha)
    post_hoc = survey.post_hoc_positive
    completed_strata = [
        stratum for stratum in survey.strata if stratum.name == "completed"
    ]
    completed_population = sum(stratum.positive for stratum in completed_strata)
    return Estimate(
        point=population_0.point + population_1.point + completed_population + post_hoc,
        lower=population_0.lower + population_1.lower + completed_population + post_hoc,
        upper=population_0.upper + population_1.upper + completed_population + post_hoc,
    )


class EstimateType(Enum):
    PROPORTION = "proportion"
    POPULATION = "population"


def stratum_f_estimator(
    survey, alpha=0.05, report: EstimateType = EstimateType.POPULATION
):
    estimate = Estimate(0, 0, 0)
    total = sum(stratum.total for stratum in survey.strata)
    for stratum in survey.strata:
        weight = stratum.total / total
        if stratum.positive == 0:
            estimate.lower += 0
        else:
            estimate.lower += (weight * stratum.positive) / (
                stratum.positive
                + (stratum.labeled - stratum.positive + 1)
                * f.ppf(
                    1 - alpha / 2,
                    2 * (stratum.labeled - stratum.positive + 1),
                    2 * stratum.positive,
                )
            )
        upper = (
            weight
            * (stratum.positive + 1)
            * f.ppf(
                1 - alpha / 2,
                2 * (stratum.positive + 1),
                2 * (stratum.labeled - stratum.positive),
            )
            / (
                (stratum.labeled - stratum.positive)
                + (stratum.positive + 1)
                * f.ppf(
                    1 - alpha / 2,
                    2 * (stratum.positive + 1),
                    2 * (stratum.labeled - stratum.positive),
                )
            )
        )
        estimate.upper += upper
    unlabeled = sum(stratum.unlabeled for stratum in survey.strata)
    positive = sum(stratum.positive for stratum in survey.strata)

    for s in survey.strata:
        s.weight = s.total / total
    p_hat = sum(
        (stratum.positive / stratum.labeled) * stratum.weight
        for stratum in survey.strata
    )
    rescaling_factor = (
        sum((s.weight**2) / s.labeled for s in survey.strata) ** 0.5
    ) / sum(s.weight / (s.labeled**0.5) for s in survey.strata)

    estimate.lower = p_hat - (p_hat - estimate.lower) * rescaling_factor
    estimate.upper = p_hat + (estimate.upper - p_hat) * rescaling_factor
    estimate.point = p_hat

    if report == "proportion":
        return estimate

    pop = estimate * unlabeled + positive

    return Estimate(
        point=int(pop.point),
        lower=int(pop.lower),
        upper=int(pop.upper),
    )


def one_stratum_f_ci(n, p, alpha=0.05):
    stratum = Stratum(name="stratum", unlabeled=1000, labeled=n, positive=p)
    return stratum_f_estimator(
        Survey(strata=[stratum], post_hoc_positive=0), alpha=alpha
    )


def one_stratum_standard_ci(n, p, alpha=0.05, method="beta"):
    conf = proportion_confint(p, n, alpha=alpha, method=method)
    return Estimate(
        point=p,
        lower=conf[0],
        upper=conf[1],
    )


def plot_comparison():
    import matplotlib.pyplot as plt
    import seaborn as sns

    ns = np.linspace(100, 10000, 50)
    ns = [int(n) for n in ns]
    p = 1

    f_ci = [one_stratum_f_ci(n, p) for n in ns]
    methods = ["binom_test", "wilson", "beta"]
    method_dfs = []
    for method in methods:
        df = pd.DataFrame(
            {
                "n": ns,
                "upper": [
                    one_stratum_standard_ci(n, p, method=method).upper for n in ns
                ],
                "method": method,
            }
        )
        method_dfs.append(df)
    f_df = pd.DataFrame(
        {
            "n": ns,
            "upper": [ci.upper for ci in f_ci],
            "method": "f",
        }
    )
    df = pd.concat([f_df] + method_dfs)
    sns.scatterplot(data=df, x="n", y="upper", hue="method")
    plt.show()


def compare_strategies():
    import matplotlib.pyplot as plt
    import seaborn as sns
    import tqdm

    labeling_strategies = ["greatest variance", "least labeled"]
    recall_methods = ["per_stratum", "aggregate"]
    step = 50
    label_budget = np.arange(0, 2000, step)
    results = []
    pb = tqdm.tqdm(
        total=len(labeling_strategies) * len(recall_methods) * len(label_budget)
    )
    db_survey = Survey.from_db()
    for strategy in labeling_strategies:
        for recall_method in recall_methods:
            survey = db_survey.copy()
            for n in label_budget:
                survey.label(n=50, method=strategy)
                recall = survey.completeness(method=recall_method)
                results.append(
                    {
                        "strategy": strategy,
                        "recall_method": recall_method,
                        "label_budget": n,
                        "recall": recall.point,
                        "lower": recall.lower,
                        "upper": recall.upper,
                    }
                )
                pb.update(1)
    df = pd.DataFrame(results)
    sns.lineplot(
        data=df, x="label_budget", y="lower", hue="strategy", style="recall_method"
    )
    plt.title("Recall vs. Label Budget")
    plt.show()

    return df


def number_of_images_per_facility():
    from cacafo.query import cafos

    total_positive_images = (
        sa.select(sa.func.count(sa.distinct(m.Image.id)))
        .select_from(m.Image)
        .join(m.ImageAnnotation)
        .join(m.Building)
        .where(m.Building.excluded_at.is_(None))
        .scalar_subquery()
    )
    total_facilities = (
        sa.select(sa.func.count()).select_from(cafos().subquery()).scalar_subquery()
    )
    return sa.select(total_positive_images / total_facilities)


def mean_facilities_per_image():
    from cacafo.query import cafos

    total_positive_images = (
        sa.select(sa.func.count(sa.distinct(m.Image.id)))
        .select_from(m.Image)
        .join(m.ImageAnnotation)
        .join(m.Building)
        .where(m.Building.excluded_at.is_(None))
        .scalar_subquery()
    )
    total_facilities = (
        sa.select(sa.func.count()).select_from(cafos().subquery()).scalar_subquery()
    )

    return sa.select(total_facilities / total_positive_images)


def estimate_population():
    survey = Survey.from_db()
    return naip_stratum_f_estimator(survey)


if __name__ == "__main__":
    print(naip_stratum_f_estimator(Survey.from_db()))
