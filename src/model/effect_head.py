"""Training-only treatment-effect estimation for CausalSeqGAN."""

from __future__ import annotations

import hashlib
import itertools
import json
import re
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import ndtr

from src.interventions import ADAPTIVITY, intervention_family

_EXCLUDED_COLUMNS = {
    "dataset_name",
    "experiment_id",
    "dependent_measure",
    "independent_measure_pair",
    "condition",
    "student_id",
    "assignment_completed",
    "normalized_student_learning",
    "task_key",
    "task_id",
    "y_binary",
    "y0_prior",
}


@dataclass(frozen=True)
class EffectHeadConfig:
    inner_seeds: tuple[int, ...] = (2026110301, 2026110302, 2026110303)
    inner_folds: int = 2
    source_mix: float = 0.5
    top_candidates: int = 10
    noise_multiplier: float = 2.0
    neural_weight: float = 0.01
    covariate_ridge: float = 0.1
    covariate_shrinkage: float = 0.5
    covariate_neural_weight: float = 0.1

    def __post_init__(self) -> None:
        if self.inner_folds < 2 or not self.inner_seeds:
            raise ValueError("Use at least two inner folds and one partition seed")
        if not 0 <= self.source_mix <= 1 or not 1 <= self.top_candidates <= 288:
            raise ValueError("Invalid calibration candidate selection settings")
        if self.noise_multiplier <= 0 or self.covariate_ridge <= 0 or self.covariate_shrinkage < 0:
            raise ValueError("Invalid calibration regularisation settings")


def training_rows_hash(rows: pd.DataFrame) -> str:
    columns = ["student_id", "task_key", "condition", "y_binary"]
    missing = set(columns) - set(rows)
    if missing:
        raise ValueError(f"Missing training columns: {sorted(missing)}")
    payload = pd.util.hash_pandas_object(rows[columns], index=False).to_numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def participant_folds(
    rows: pd.DataFrame, *, folds: int, seed: int, min_fold_arm: int = 4, attempts: int = 128
) -> np.ndarray:
    """Partition participants while retaining both arms within every task."""
    participants = rows.student_id.astype(str)
    unique = participants.drop_duplicates().to_numpy()
    best, best_score = None, float("inf")
    for attempt in range(attempts):
        permutation = np.random.default_rng(seed + 104729 * attempt).permutation(unique)
        mapping = {participant: index % folds for index, participant in enumerate(permutation)}
        assignment = participants.map(mapping).to_numpy(dtype=np.int64)
        counts = (
            rows.assign(_fold=assignment, _participant=participants)
            .groupby(["task_key", "condition", "_fold"])
            ._participant.nunique()
            .unstack(fill_value=0)
            .reindex(columns=range(folds), fill_value=0)
        )
        proportions = counts.div(counts.sum(axis=1), axis=0)
        size = np.bincount(assignment, minlength=folds).astype(float)
        score = float(
            np.square(proportions - 1.0 / folds).mean().mean()
            + np.square(size / size.sum() - 1.0 / folds).mean()
        )
        if counts.to_numpy().min() >= min_fold_arm and score < best_score:
            best, best_score = assignment, score
    if best is None:
        raise ValueError("No participant partition satisfies the task-arm sample requirement")
    return best


def _task_statistics(
    tasks: pd.DataFrame,
    train: pd.DataFrame,
    priors: pd.DataFrame | None = None,
    validation: pd.DataFrame | None = None,
) -> pd.DataFrame:
    prior = None if priors is None else priors.set_index("task_key")
    records = []
    for task in tasks.itertuples(index=False):
        subset = train[train.task_key.eq(task.task_key)]
        y0 = subset.loc[subset.condition.eq(0), "y_binary"].to_numpy(float)
        y1 = subset.loc[subset.condition.eq(1), "y_binary"].to_numpy(float)
        if not len(y0) or not len(y1):
            raise ValueError(f"Missing training arm for task {task.task_key}")
        p0, p1 = float(y0.mean()), float(y1.mean())
        row = dict(
            task_key=task.task_key,
            experiment_id=str(task.experiment_id),
            independent_measure_pair=task.independent_measure_pair,
            dependent_measure=task.dependent_measure,
            n0_train=len(y0),
            n1_train=len(y1),
            train_ate=p1 - p0,
            train_pooled_value=float(subset.y_binary.mean()),
            sampling_variance=max(1e-8, p0 * (1 - p0) / len(y0) + p1 * (1 - p1) / len(y1)),
            scm_ate=0.0,
        )
        if prior is not None:
            prediction = prior.loc[task.task_key]
            row["scm_ate"] = float(prediction.model_value_t1 - prediction.model_value_t0)
        if validation is not None:
            heldout = validation[validation.task_key.eq(task.task_key)]
            a = heldout.loc[heldout.condition.eq(0), "y_binary"].to_numpy(float)
            b = heldout.loc[heldout.condition.eq(1), "y_binary"].to_numpy(float)
            if not len(a) or not len(b):
                raise ValueError(f"Missing validation arm for task {task.task_key}")
            row["inner_reference_ate"] = float(b.mean() - a.mean())
        records.append(row)
    frame = pd.DataFrame(records)
    frame["family"] = frame.independent_measure_pair.map(intervention_family)
    return frame


def _features(frame: pd.DataFrame) -> dict[str, Any]:
    data = frame.reset_index(drop=True)
    variance = data.sampling_variance.to_numpy()
    direct, neural = data.train_ate.to_numpy(), data.scm_ate.to_numpy()
    weights = 1.0 / data.groupby("experiment_id").task_key.transform("count").to_numpy()
    weights /= weights.sum()
    anchors = dict(
        zero=np.zeros(len(data)),
        global_=np.repeat(np.sum(weights * direct), len(data)),
        family_median=data.groupby("family").train_ate.transform("median").to_numpy(),
    )
    anchors["global"] = anchors.pop("global_")
    for column, name, precision in (
        ("family", "family", False),
        ("independent_measure_pair", "pair_precision", True),
        ("dependent_measure", "outcome_precision", True),
    ):
        anchor = np.zeros(len(data))
        for indices in data.groupby(column).groups.values():
            index = np.array(list(indices))
            local = (
                weights[index] / (variance[index] + 1e-5) if precision else weights[index].copy()
            )
            local /= local.sum()
            anchor[index] = np.sum(local * direct[index])
        anchors[name] = anchor
    return dict(
        data=data,
        direct=direct,
        variance=variance,
        anchors=anchors,
        centered_neural=neural - np.sum(weights * neural),
    )


def _candidate_bank() -> list[dict[str, Any]]:
    candidates = []
    for anchor in (
        "zero",
        "global",
        "family",
        "family_median",
        "pair_precision",
        "outcome_precision",
    ):
        candidates.extend(
            dict(anchor=anchor, method="fixed", weight=w) for w in (0, 0.1, 0.25, 0.5, 0.75, 1)
        )
        candidates.extend(
            dict(anchor=anchor, method="normal", tau=t)
            for t in (0.00001, 0.0001, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.02)
        )
        candidates.extend(
            dict(anchor=anchor, method="soft", threshold=t) for t in (0, 0.25, 0.5, 1, 1.5, 2, 3)
        )
        candidates.extend(
            dict(anchor=anchor, method="spike_median", tau=t, spike=p, predictive=v)
            for t, p, v in itertools.product((0.001, 0.005, 0.02), (0.5, 0.8, 0.95), (0, 1, 2))
        )
    return candidates


def _effect(
    features: dict[str, Any],
    candidate: dict[str, Any],
    *,
    gamma: float = 0.0,
    noise_multiplier: float = 1.0,
) -> np.ndarray:
    anchor = features["anchors"][candidate["anchor"]] + gamma * features["centered_neural"]
    direct, variance = features["direct"], features["variance"] * noise_multiplier
    method = candidate["method"]
    if method == "fixed":
        return anchor + candidate["weight"] * (direct - anchor)
    if method == "normal":
        return anchor + candidate["tau"] / (candidate["tau"] + variance) * (direct - anchor)
    if method == "soft":
        return anchor + np.sign(direct - anchor) * np.maximum(
            abs(direct - anchor) - candidate["threshold"] * np.sqrt(variance), 0
        )
    if method != "spike_median":
        raise ValueError(f"Unknown effect estimator: {method}")
    tau, spike = candidate["tau"], candidate["spike"]
    log_spike = np.log(spike) - 0.5 * np.log(variance) - 0.5 * (direct - anchor) ** 2 / variance
    log_slab = (
        np.log(1 - spike)
        - 0.5 * np.log(variance + tau)
        - 0.5 * (direct - anchor) ** 2 / (variance + tau)
    )
    probability = 1 / (1 + np.exp(np.clip(log_slab - log_spike, -700, 700)))
    slab_mean = anchor + tau / (tau + variance) * (direct - anchor)
    spike_sd = np.sqrt(np.maximum(1e-12, candidate["predictive"] * variance))
    slab_sd = np.sqrt(tau * variance / (tau + variance) + candidate["predictive"] * variance)
    lower = np.minimum(anchor - 8 * spike_sd, slab_mean - 8 * slab_sd)
    upper = np.maximum(anchor + 8 * spike_sd, slab_mean + 8 * slab_sd)
    for _ in range(24):
        middle = (lower + upper) / 2
        cdf = probability * ndtr((middle - anchor) / spike_sd) + (1 - probability) * ndtr(
            (middle - slab_mean) / slab_sd
        )
        lower, upper = np.where(cdf < 0.5, middle, lower), np.where(cdf >= 0.5, middle, upper)
    return (lower + upper) / 2


def _experiment_weights(data: pd.DataFrame) -> np.ndarray:
    counts = (
        data[["task_key", "experiment_id"]]
        .drop_duplicates()
        .groupby("experiment_id")
        .task_key.count()
    )
    return 1 / data.experiment_id.map(counts).to_numpy()


def _covariate_transform(all_rows: pd.DataFrame, train: pd.DataFrame) -> np.ndarray:
    columns, all_parts, train_parts = [], [], []
    has_prior = any(
        re.sub(r"[^a-z0-9]+", "", str(c).lower()) == "priorpercentcorrect" for c in all_rows
    )
    for column in all_rows:
        normalized = re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower()).strip("_")
        if column in _EXCLUDED_COLUMNS or normalized == "id" or normalized.endswith("id"):
            continue
        if column == "student_prior_average_correctness" and has_prior:
            continue
        if column == "opportunity_zone":
            mapping = {"yes": 1.0, "true": 1.0, "1": 1.0, "no": 0.0, "false": 0.0, "0": 0.0}
            whole = all_rows[column].astype(str).str.lower().map(mapping)
            fitting = train[column].astype(str).str.lower().map(mapping)
        else:
            whole = pd.to_numeric(all_rows[column], errors="coerce")
            fitting = pd.to_numeric(train[column], errors="coerce")
        if fitting.notna().sum() == 0:
            continue
        median = float(fitting.median())
        if float(fitting.fillna(median).astype(float).std(ddof=1)) < 1e-12:
            continue
        columns.append(column)
        all_parts.append(whole.astype(float))
        train_parts.append(fitting.astype(float))
    if not columns:
        raise ValueError("No eligible pre-treatment covariates")
    whole, fitting = pd.concat(all_parts, axis=1), pd.concat(train_parts, axis=1)
    whole.columns = fitting.columns = columns
    median = fitting.median(axis=0).fillna(0.0)
    mean = fitting.fillna(median).mean(axis=0)
    sd = fitting.fillna(median).std(axis=0).replace(0.0, 1.0).fillna(1.0)
    return ((whole.fillna(median) - mean) / sd).clip(-8.0, 8.0).to_numpy(dtype=np.float32)


def _covariate_predictions(
    train: pd.DataFrame, evaluation: pd.DataFrame, priors: pd.DataFrame, config: EffectHeadConfig
) -> pd.DataFrame:
    rows = pd.concat([train, evaluation], ignore_index=True)
    fitting, heldout = rows.iloc[: len(train)], rows.iloc[len(train) :]
    covariates = _covariate_transform(rows, fitting)
    prior = priors.set_index("task_key")
    records = []
    for key, task in fitting.groupby("task_key", sort=False):
        target = heldout[heldout.task_key.eq(key)]
        x, x_eval = covariates[task.index].astype(float), covariates[target.index].astype(float)
        outcome, treatment = task.y_binary.to_numpy(float), task.condition.to_numpy(float)
        mean = x.mean(0)
        xc, yc, tc = x - mean, outcome - outcome.mean(), treatment - treatment.mean()
        xx, xy, tx, ty, tt = xc.T @ xc, xc.T @ yc, tc @ xc, tc @ yc, tc @ tc
        inverse = np.linalg.pinv(xx + config.covariate_ridge * np.eye(xx.shape[0]))
        tau = float((ty - tx @ inverse @ xy) / max(float(tt - tx @ inverse @ tx), 1e-8))
        shift = x_eval.mean(0) - mean
        records.append(
            dict(
                task_key=key,
                experiment_id=str(task.experiment_id.iloc[0]),
                family=intervention_family(task.independent_measure_pair.iloc[0]),
                neural_prior=float(
                    prior.loc[key, "model_value_t1"] - prior.loc[key, "model_value_t0"]
                ),
                train_direct=float(outcome[treatment == 1].mean() - outcome[treatment == 0].mean()),
                tau_adjusted=tau,
                b0=float(shift @ (inverse @ xy)),
                bt=float(shift @ (inverse @ tx)),
                pooled=float(outcome.mean()),
                p_t=float(treatment.mean()),
            )
        )
    stats = pd.DataFrame(records)
    weight = 1 / stats.groupby("experiment_id").task_key.transform("count").to_numpy(float)
    center = float(np.average(stats.train_direct, weights=weight))
    neural_center = float(np.average(stats.neural_prior, weights=weight))
    target = center + config.covariate_neural_weight * (
        stats.neural_prior.to_numpy() - neural_center
    )
    effect = (stats.tau_adjusted.to_numpy() + config.covariate_shrinkage * target) / (
        1 + config.covariate_shrinkage
    )
    mu0 = (
        stats.pooled.to_numpy()
        + stats.b0.to_numpy()
        - stats.bt.to_numpy() * effect
        - stats.p_t.to_numpy() * effect
    )
    result = stats[["task_key", "family"]].copy()
    result["model_value_t0"], result["model_value_t1"] = (
        np.clip(mu0, 0, 1),
        np.clip(mu0 + effect, 0, 1),
    )
    return result


class CausalSeqGANEffectHead:
    """Fit and apply task-level effect heads using training participants."""

    def __init__(self, config: EffectHeadConfig | None = None):
        self.config = EffectHeadConfig() if config is None else config
        self._state: dict[str, Any] | None = None

    def fit(
        self,
        tasks: pd.DataFrame,
        train_rows: pd.DataFrame,
        *,
        source: str,
        outer_fold: int,
        cache_dir: str | Path | None = None,
    ) -> CausalSeqGANEffectHead:
        train = train_rows.copy().reset_index(drop=True)
        digest = training_rows_hash(train)
        partition = json.dumps(
            dict(
                training_rows_sha256=digest,
                seeds=self.config.inner_seeds,
                folds=self.config.inner_folds,
            ),
            sort_keys=True,
        ).encode()
        cache_key = hashlib.sha256(partition).hexdigest()[:16]
        cache = (
            None
            if cache_dir is None
            else Path(cache_dir) / f"{source}_{outer_fold}_{cache_key}.csv"
        )
        if cache is not None and cache.is_file():
            inner = pd.read_csv(cache, dtype={"experiment_id": str})
        else:
            parts = []
            for repeat, seed in enumerate(self.config.inner_seeds):
                assignment = participant_folds(
                    train, folds=self.config.inner_folds, seed=seed + outer_fold * 10007
                )
                for fold in range(self.config.inner_folds):
                    fitting, validation = (
                        train.loc[assignment != fold],
                        train.loc[assignment == fold],
                    )
                    if set(fitting.student_id.astype(str)) & set(validation.student_id.astype(str)):
                        raise ValueError("Overlapping inner training and validation participants")
                    frame = _task_statistics(tasks, fitting, validation=validation)
                    frame["repeat"], frame["inner_fold"] = repeat, fold
                    parts.append(frame)
            inner = pd.concat(parts, ignore_index=True)
            if cache is not None:
                cache.parent.mkdir(parents=True, exist_ok=True)
                inner.to_csv(cache, index=False)
        bank = _candidate_bank()
        scores: dict[str, list[np.ndarray]] = {family: [] for family in inner.family.unique()}
        for _, group in inner.groupby(["repeat", "inner_fold"]):
            features = _features(group)
            data = features["data"]
            prediction = np.array([_effect(features, candidate) for candidate in bank]).T
            error = abs(prediction - data.inner_reference_ate.to_numpy()[:, None])
            for family, indices in data.groupby("family").groups.items():
                index = np.array(list(indices))
                weights = _experiment_weights(data.iloc[index])
                weights /= weights.sum()
                scores[family].append(weights @ error[index])
        family_scores = {family: np.array(values).mean(axis=0) for family, values in scores.items()}
        source_score = np.mean(list(family_scores.values()), axis=0)
        chosen = {
            family: list(
                map(
                    int,
                    np.argsort(
                        (1 - self.config.source_mix) * score
                        + self.config.source_mix * source_score,
                        kind="stable",
                    )[: self.config.top_candidates],
                )
            )
            for family, score in family_scores.items()
        }
        self._state = dict(
            source=source,
            outer_fold=int(outer_fold),
            training_rows_sha256=digest,
            selected_candidate_ids=chosen,
            candidate_bank=bank,
            configuration=asdict(self.config),
            source_mix=self.config.source_mix,
            topk=self.config.top_candidates,
            noise_multiplier=self.config.noise_multiplier,
            neural_gamma=self.config.neural_weight,
            inner_seeds=list(self.config.inner_seeds),
            inner_folds=self.config.inner_folds,
        )
        return self

    def predict(
        self,
        tasks: pd.DataFrame,
        train_rows: pd.DataFrame,
        evaluation_rows: pd.DataFrame,
        raw_predictions: pd.DataFrame,
    ) -> pd.DataFrame:
        state = self.state_dict()
        if training_rows_hash(train_rows) != state["training_rows_sha256"]:
            raise ValueError("Training rows differ from the fitted effect-head state")
        required = {"task_key", "model_value_t0", "model_value_t1"}
        if not required <= set(raw_predictions) or raw_predictions.task_key.duplicated().any():
            raise ValueError("Pass one pair of neural arm probabilities for every task")
        if set(raw_predictions.task_key) != set(tasks.task_key):
            raise ValueError("Neural predictions must cover every task in the fitted source")
        prior = raw_predictions[["task_key", "model_value_t0", "model_value_t1"]]
        statistics = _task_statistics(tasks, train_rows, prior)
        features = _features(statistics)
        effect = np.zeros(len(statistics))
        assigned = np.zeros(len(statistics), dtype=bool)
        for family, indices in state["selected_candidate_ids"].items():
            mask = statistics.family.eq(family).to_numpy()
            effect[mask] = np.mean(
                [
                    _effect(
                        features,
                        state["candidate_bank"][index],
                        gamma=state["neural_gamma"],
                        noise_multiplier=state["noise_multiplier"],
                    )[mask]
                    for index in indices
                ],
                axis=0,
            )
            assigned |= mask
        if not assigned.all():
            raise ValueError("A task family has no fitted calibration configuration")
        pooled = statistics.train_pooled_value.to_numpy()
        pi1 = (statistics.n1_train / (statistics.n0_train + statistics.n1_train)).to_numpy()
        pi0 = 1 - pi1
        effect = np.clip(
            effect,
            np.maximum((pooled - 1) / pi1, -pooled / pi0),
            np.minimum(pooled / pi1, (1 - pooled) / pi0),
        )
        result = statistics[["task_key", "family"]].copy()
        result["model_value_t0"], result["model_value_t1"] = (
            pooled - pi1 * effect,
            pooled + pi0 * effect,
        )
        if result.family.eq(ADAPTIVITY).any():
            adjusted = _covariate_predictions(
                train_rows, evaluation_rows, prior, self.config
            ).set_index("task_key")
            mask = result.family.eq(ADAPTIVITY)
            for column in ("model_value_t0", "model_value_t1"):
                result.loc[mask, column] = result.loc[mask, "task_key"].map(adjusted[column])
        probabilities = result[["model_value_t0", "model_value_t1"]].to_numpy(float)
        if (
            not np.isfinite(probabilities).all()
            or ((probabilities < 0) | (probabilities > 1)).any()
        ):
            raise ValueError("Invalid calibrated arm probabilities")
        return result

    def state_dict(self) -> dict[str, Any]:
        if self._state is None:
            raise ValueError("Fit or load the effect head before prediction")
        return deepcopy(self._state)

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> CausalSeqGANEffectHead:
        required = {
            "source",
            "outer_fold",
            "training_rows_sha256",
            "selected_candidate_ids",
            "candidate_bank",
            "source_mix",
            "topk",
            "noise_multiplier",
            "neural_gamma",
        }
        if not required <= set(state):
            raise ValueError(f"Missing fitted-state entries: {sorted(required - set(state))}")
        values = state.get(
            "configuration",
            dict(
                inner_seeds=state.get("inner_seeds", EffectHeadConfig().inner_seeds),
                inner_folds=state.get("inner_folds", 2),
                source_mix=state["source_mix"],
                top_candidates=state["topk"],
                noise_multiplier=state["noise_multiplier"],
                neural_weight=state["neural_gamma"],
            ),
        )
        if "inner_seeds" in values:
            values = {**values, "inner_seeds": tuple(values["inner_seeds"])}
        instance = cls(EffectHeadConfig(**values))
        instance._state = deepcopy(state)
        return instance

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.state_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> CausalSeqGANEffectHead:
        return cls.from_state(json.loads(Path(path).read_text(encoding="utf-8")))
