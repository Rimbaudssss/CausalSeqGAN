"""Participant-disjoint training and effect prediction on randomised trials."""

from __future__ import annotations

import argparse
import hashlib
import math
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from torch.utils.data import WeightedRandomSampler

from src.baselines import load_base_model_from_checkpoint
from src.config import CAUSALSEQGAN_ID, CAUSALSEQGAN_NAME, MODEL_KNOBS
from src.data import TrajectoryBatch, TrajectoryDataset, compute_lengths
from src.data_rct import ID_COLS
from src.training import (
    _make_dataloader,
    _set_seed,
    _train_adversarial_generator,
    _train_crn,
    _train_diffusion,
    _train_timegan,
    _train_vae,
    latest_checkpoint,
)

NULL_MODEL = "zero_effect"


STATISTICAL_MODELS = ["difference_in_means", "ancova", "aipw", "tlearner_rf", "xlearner_rf"]

METHOD_LABEL = {
    NULL_MODEL: "Zero effect",
    CAUSALSEQGAN_ID: CAUSALSEQGAN_NAME,
    "difference_in_means": "Difference in means",
    "ancova": "ANCOVA",
    "aipw": "AIPW",
    "tlearner_rf": "T-learner (RF)",
    "xlearner_rf": "X-learner (RF)",
    "rcgan": "RCGAN",
    "crn": "CRN",
    "vae": "VAE",
    "diffusion": "Diffusion",
    "timegan": "TimeGAN",
}


def method_family(model: str) -> str:
    if model == NULL_MODEL:
        return "null_reference"
    if model in STATISTICAL_MODELS:
        return "statistical_causal"
    return "generative_model"


def administrative_identifier(column: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower()).strip("_")
    return normalized == "id" or normalized.endswith("_id") or normalized.endswith("id")


def numeric_feature(column: str, values: pd.Series) -> pd.Series:
    if column == "opportunity_zone":
        mapping = {"yes": 1.0, "true": 1.0, "1": 1.0, "no": 0.0, "false": 0.0, "0": 0.0}
        return values.astype(str).str.lower().map(mapping)
    return pd.to_numeric(values, errors="coerce")


def fit_feature_transform(
    all_rows: pd.DataFrame,
    train_rows: pd.DataFrame,
    *,
    clip: float,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    blocked = ID_COLS | {"task_key", "task_id", "y_binary", "y0_prior"}
    has_source_prior = any(
        re.sub(r"[^a-z0-9]+", "", str(column).lower()) == "priorpercentcorrect"
        for column in all_rows.columns
    )
    included: list[str] = []
    excluded: dict[str, str] = {}
    all_parts: list[pd.Series] = []
    train_parts: list[pd.Series] = []
    for column in all_rows.columns:
        if column in blocked:
            excluded[column] = "identifier, treatment, or outcome"
            continue
        if administrative_identifier(column):
            excluded[column] = "administrative identifier"
            continue
        if column == "student_prior_average_correctness" and has_source_prior:
            excluded[column] = "duplicate prior-percent-correct feature"
            continue
        all_values = numeric_feature(column, all_rows[column])
        train_values = numeric_feature(column, train_rows[column])
        if train_values.notna().sum() == 0:
            excluded[column] = "non-numeric or entirely missing in training fold"
            continue
        median = float(train_values.median())
        train_filled = train_values.fillna(median).astype(float)
        if float(train_filled.std(ddof=1)) < 1e-12:
            excluded[column] = "constant in training fold"
            continue
        included.append(column)
        all_parts.append(all_values.astype(float))
        train_parts.append(train_values.astype(float))
    if not included:
        raise RuntimeError("No eligible pre-treatment features remain after the leakage audit.")

    all_frame = pd.concat(all_parts, axis=1)
    train_frame = pd.concat(train_parts, axis=1)
    all_frame.columns = included
    train_frame.columns = included
    median = train_frame.median(axis=0).fillna(0.0)
    filled_train = train_frame.fillna(median)
    mean = filled_train.mean(axis=0)
    std = filled_train.std(axis=0).replace(0.0, 1.0).fillna(1.0)
    transformed = ((all_frame.fillna(median) - mean) / std).clip(-float(clip), float(clip))
    metadata = {
        "feature_columns": included,
        "excluded_columns": excluded,
        "median": median.to_dict(),
        "mean": mean.to_dict(),
        "std": std.to_dict(),
        "standardized_clip": float(clip),
        "fit_scope": "training folds only",
    }
    return transformed.to_numpy(dtype=np.float32), included, metadata


def assign_participant_folds(
    rows: pd.DataFrame,
    *,
    folds: int,
    seed: int,
    min_fold_arm: int,
    attempts: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    participants = rows["student_id"].astype(str)
    unique_participants = participants.drop_duplicates().to_numpy()
    best_assignment: np.ndarray | None = None
    best_score = float("inf")
    best_minimum = -1
    best_attempt = -1
    for attempt in range(int(attempts)):
        rng = np.random.default_rng(int(seed) + 104_729 * attempt)
        permutation = rng.permutation(unique_participants)
        mapping = {participant: index % int(folds) for index, participant in enumerate(permutation)}
        assignment = participants.map(mapping).to_numpy(dtype=np.int64)
        counts = (
            rows.assign(_fold=assignment, _participant=participants)
            .groupby(["task_key", "condition", "_fold"])["_participant"]
            .nunique()
            .unstack(fill_value=0)
            .reindex(columns=range(int(folds)), fill_value=0)
        )
        minimum = int(counts.to_numpy().min())
        proportions = counts.div(counts.sum(axis=1), axis=0)
        imbalance = float(np.square(proportions - 1.0 / float(folds)).mean().mean())
        fold_sizes = np.bincount(assignment, minlength=int(folds)).astype(float)
        size_imbalance = float(np.square(fold_sizes / fold_sizes.sum() - 1.0 / float(folds)).mean())
        score = imbalance + size_imbalance
        if minimum >= int(min_fold_arm) and score < best_score:
            best_assignment = assignment
            best_score = score
            best_minimum = minimum
            best_attempt = attempt
    if best_assignment is None:
        raise RuntimeError(
            f"No {folds}-fold participant partition reached {min_fold_arm} participants in every task arm."
        )
    digest_rows = sorted(
        f"{participant}:{int(fold)}"
        for participant, fold in zip(
            participants.drop_duplicates(),
            best_assignment[pd.Series(participants).drop_duplicates().index],
        )
    )
    digest = hashlib.sha256("\n".join(digest_rows).encode("utf-8")).hexdigest()
    return best_assignment, {
        "fold_seed": int(seed),
        "folds": int(folds),
        "attempt": int(best_attempt),
        "minimum_task_arm_participants": int(best_minimum),
        "imbalance_score": float(best_score),
        "participant_count": int(len(unique_participants)),
        "assignment_sha256": digest,
    }


def split_audit(
    rows: pd.DataFrame, assignment: np.ndarray, *, dataset: str, folds: int
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    rows["student_id"].astype(str)
    for fold in range(int(folds)):
        train = rows.loc[assignment != fold]
        test = rows.loc[assignment == fold]
        train_ids = set(train["student_id"].astype(str))
        test_ids = set(test["student_id"].astype(str))
        overlap = len(train_ids & test_ids)
        if overlap:
            raise AssertionError(
                f"Participant leakage in dataset={dataset}, fold={fold}: {overlap}"
            )
        for task_key, task_rows in rows.groupby("task_key", sort=False):
            train_task = train[train["task_key"].eq(task_key)]
            test_task = test[test["task_key"].eq(task_key)]
            records.append(
                {
                    "dataset": dataset,
                    "fold": int(fold),
                    "task_key": task_key,
                    "n_total": int(len(task_rows)),
                    "n_train": int(len(train_task)),
                    "n_eval": int(len(test_task)),
                    "n0_train": int(train_task["condition"].eq(0).sum()),
                    "n1_train": int(train_task["condition"].eq(1).sum()),
                    "n0_eval": int(test_task["condition"].eq(0).sum()),
                    "n1_eval": int(test_task["condition"].eq(1).sum()),
                    "train_participants": int(train_task["student_id"].astype(str).nunique()),
                    "eval_participants": int(test_task["student_id"].astype(str).nunique()),
                    "participant_overlap": int(overlap),
                }
            )
    audit = pd.DataFrame(records)
    if int(audit[["n0_eval", "n1_eval"]].min().min()) <= 0:
        raise AssertionError("At least one cross-fitting fold is missing a randomized arm.")
    return audit


def make_contextual_batch(
    rows: pd.DataFrame,
    x_all: np.ndarray,
    *,
    device: torch.device | None = None,
) -> TrajectoryBatch:
    indices = rows.index.to_numpy()
    n_rows = len(rows)
    x = torch.as_tensor(x_all[indices], dtype=torch.float32)
    a = torch.as_tensor(
        (rows["task_id"].to_numpy(dtype=np.int64) + 1)[:, None].copy(), dtype=torch.long
    )
    treatment = torch.as_tensor(
        rows["condition"].to_numpy(dtype=np.int64)[:, None].copy(), dtype=torch.long
    )
    outcome = torch.as_tensor(
        rows["y_binary"].to_numpy(dtype=np.float32)[:, None].copy(), dtype=torch.float32
    )
    mask = torch.ones((n_rows, 1), dtype=torch.float32)
    batch = TrajectoryBatch(
        X=x, A=a, T=treatment, Y=outcome, mask=mask, lengths=compute_lengths(mask)
    )
    return batch.to(device) if device is not None else batch


def experiment_task_arm_sample_weights(rows: pd.DataFrame) -> torch.Tensor:
    """Give each experiment equal mass, then each task and randomized arm equal mass."""
    required = {"experiment_id", "task_key", "condition"}
    missing = required - set(rows.columns)
    if missing:
        raise KeyError(f"Cannot balance training rows without columns: {sorted(missing)}")
    task_counts = (
        rows[["experiment_id", "task_key"]]
        .drop_duplicates()
        .groupby("experiment_id")["task_key"]
        .size()
    )
    cell_counts = rows.groupby(["experiment_id", "task_key", "condition"])["condition"].transform(
        "size"
    )
    tasks_per_experiment = rows["experiment_id"].map(task_counts).astype(float)
    weights = 1.0 / (tasks_per_experiment * cell_counts.astype(float))
    weights *= float(len(weights)) / float(weights.sum())
    if not np.isfinite(weights).all() or bool((weights <= 0.0).any()):
        raise RuntimeError("Experiment-task-arm balancing produced invalid weights")
    return torch.as_tensor(weights.to_numpy(dtype=np.float64).copy(), dtype=torch.double)


@torch.inference_mode()
def estimate_contextual_values(
    generator: Any,
    rows: pd.DataFrame,
    x_all: np.ndarray,
    *,
    seed: int,
    model_index: int,
    mc_samples: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    totals = {0: 0.0, 1: 0.0}
    n_total = 0
    for start in range(0, len(rows), int(batch_size)):
        chunk = rows.iloc[start : start + int(batch_size)]
        batch = make_contextual_batch(chunk, x_all)
        sums = {
            0: torch.zeros(len(chunk), device=device),
            1: torch.zeros(len(chunk), device=device),
        }
        for sample in range(int(mc_samples)):
            draw_seed = int(seed) * 100_000 + int(model_index) * 1_000 + int(sample)
            for arm in (0, 1):
                torch.manual_seed(draw_seed)
                output = generator.rollout(batch, do_t={0: arm})
                sums[arm] += output["Y_prob"][:, 0].to(device).float()
        for arm in (0, 1):
            totals[arm] += float((sums[arm] / float(mc_samples)).sum().item())
        n_total += len(chunk)
    return {
        "model_value_t0": totals[0] / max(1, n_total),
        "model_value_t1": totals[1] / max(1, n_total),
    }


def fold_result(
    *,
    dataset: str,
    fold: int,
    seed: int,
    model: str,
    task: pd.Series,
    train_rows: pd.DataFrame,
    eval_rows: pd.DataFrame,
    values: dict[str, float],
    elapsed_seconds: float,
) -> dict[str, Any]:
    reference = full_reference(eval_rows)
    mu0 = float(values["model_value_t0"])
    mu1 = float(values["model_value_t1"])
    ate_hat = mu1 - mu0
    true_mu0 = float(reference["true_value_t0"])
    true_mu1 = float(reference["true_value_t1"])
    true_ate = float(reference["true_ate"])
    selected_arm = int(mu1 >= mu0)
    selected_true_value = true_mu1 if selected_arm else true_mu0
    return {
        "dataset": dataset,
        "fold": int(fold),
        "seed": int(seed),
        "model": model,
        "label": METHOD_LABEL[model],
        "method_family": method_family(model),
        "task_key": task["task_key"],
        "experiment_id": task["experiment_id"],
        "independent_measure_pair": task["independent_measure_pair"],
        "dependent_measure": task["dependent_measure"],
        "category": task["category"],
        "n_train": int(len(train_rows)),
        "n0_train": int(train_rows["condition"].eq(0).sum()),
        "n1_train": int(train_rows["condition"].eq(1).sum()),
        "n_eval": int(len(eval_rows)),
        "n0_eval": int(eval_rows["condition"].eq(0).sum()),
        "n1_eval": int(eval_rows["condition"].eq(1).sum()),
        "model_value_t0": mu0,
        "model_value_t1": mu1,
        "fold_true_value_t0": true_mu0,
        "fold_true_value_t1": true_mu1,
        "fold_true_ate": true_ate,
        "fold_true_ate_se": float(reference["true_ate_se"]),
        "fold_ate_error": ate_hat - true_ate,
        "fold_ate_abs_err": abs(ate_hat - true_ate),
        "fold_ate_squared_err": (ate_hat - true_ate) ** 2,
        "fold_policy_value_abs_err": abs((mu1 if selected_arm else mu0) - selected_true_value),
        "fold_policy_regret": max(0.0, max(true_mu0, true_mu1) - selected_true_value),
        "elapsed_seconds": float(elapsed_seconds),
    }


def full_reference(rows: pd.DataFrame) -> dict[str, float | int | bool]:
    outcome = rows["y_binary"].to_numpy(float)
    treatment = rows["condition"].to_numpy(int)
    arm0 = outcome[treatment == 0]
    arm1 = outcome[treatment == 1]
    mu0 = float(arm0.mean())
    mu1 = float(arm1.mean())
    ate = mu1 - mu0
    se = math.sqrt(mu0 * (1.0 - mu0) / len(arm0) + mu1 * (1.0 - mu1) / len(arm1))
    low = max(-1.0, ate - 1.96 * se)
    high = min(1.0, ate + 1.96 * se)
    return {
        "n_eval": int(len(rows)),
        "n0_eval": int(len(arm0)),
        "n1_eval": int(len(arm1)),
        "true_value_t0": mu0,
        "true_value_t1": mu1,
        "true_ate": ate,
        "true_ate_se": se,
        "true_ate_ci_low": low,
        "true_ate_ci_high": high,
        "effect_distinguishable": bool(low > 0.0 or high < 0.0),
    }


def stable_ridge_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    *,
    alpha: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    model = Ridge(alpha=float(alpha))
    model.fit(x_train, y_train)
    return np.asarray(model.predict(x_train), dtype=float), np.asarray(
        model.predict(x_eval), dtype=float
    )


def estimate_statistical_values(
    method: str,
    train_rows: pd.DataFrame,
    eval_rows: pd.DataFrame,
    x_all: np.ndarray,
    *,
    seed: int,
) -> dict[str, float]:
    train_index = train_rows.index.to_numpy()
    eval_index = eval_rows.index.to_numpy()
    x_train = x_all[train_index]
    x_eval = x_all[eval_index]
    treatment = train_rows["condition"].to_numpy(int)
    outcome = train_rows["y_binary"].to_numpy(float)
    if np.sum(treatment == 0) < 2 or np.sum(treatment == 1) < 2:
        raise ValueError(
            "Statistical baselines require at least two training observations per arm."
        )

    if method == "difference_in_means":
        return {
            "model_value_t0": float(np.mean(outcome[treatment == 0])),
            "model_value_t1": float(np.mean(outcome[treatment == 1])),
        }

    if method == "ancova":
        design = np.column_stack([treatment, x_train])
        model = Ridge(alpha=1.0)
        model.fit(design, outcome)
        mu0 = np.clip(model.predict(np.column_stack([np.zeros(len(x_eval)), x_eval])), 0.0, 1.0)
        mu1 = np.clip(model.predict(np.column_stack([np.ones(len(x_eval)), x_eval])), 0.0, 1.0)
        return {"model_value_t0": float(mu0.mean()), "model_value_t1": float(mu1.mean())}

    if method == "aipw":
        _, m0_eval = stable_ridge_predict(x_train[treatment == 0], outcome[treatment == 0], x_eval)
        _, m1_eval = stable_ridge_predict(x_train[treatment == 1], outcome[treatment == 1], x_eval)
        full_m0_train = np.clip(
            Ridge(alpha=1.0).fit(x_train[treatment == 0], outcome[treatment == 0]).predict(x_train),
            0.0,
            1.0,
        )
        full_m1_train = np.clip(
            Ridge(alpha=1.0).fit(x_train[treatment == 1], outcome[treatment == 1]).predict(x_train),
            0.0,
            1.0,
        )
        try:
            propensity = LogisticRegression(
                C=1.0, max_iter=1000, solver="lbfgs", random_state=int(seed)
            )
            propensity.fit(x_train, treatment)
            e_hat = propensity.predict_proba(x_train)[:, 1]
        except ValueError:
            e_hat = np.full(len(x_train), np.mean(treatment), dtype=float)
        e_hat = np.clip(e_hat, 0.05, 0.95)
        residual = treatment * (outcome - full_m1_train) / e_hat - (1 - treatment) * (
            outcome - full_m0_train
        ) / (1.0 - e_hat)
        tau = float(
            np.mean(np.clip(m1_eval, 0.0, 1.0) - np.clip(m0_eval, 0.0, 1.0)) + np.mean(residual)
        )
        tau = float(np.clip(tau, -1.0, 1.0))
        center = float(np.mean((np.clip(m0_eval, 0.0, 1.0) + np.clip(m1_eval, 0.0, 1.0)) / 2.0))
        mu0 = float(np.clip(center - tau / 2.0, 0.0, 1.0))
        mu1 = float(np.clip(center + tau / 2.0, 0.0, 1.0))
        return {"model_value_t0": mu0, "model_value_t1": mu1}

    if method == "tlearner_rf":
        predictions: dict[int, np.ndarray] = {}
        for arm in (0, 1):
            selected = treatment == arm
            min_leaf = max(2, min(8, int(np.sum(selected) // 12)))
            model = RandomForestRegressor(
                n_estimators=150,
                min_samples_leaf=min_leaf,
                max_features="sqrt",
                random_state=int(seed) + 101 * arm,
                n_jobs=1,
            )
            model.fit(x_train[selected], outcome[selected])
            predictions[arm] = np.clip(model.predict(x_eval), 0.0, 1.0)
        return {
            "model_value_t0": float(predictions[0].mean()),
            "model_value_t1": float(predictions[1].mean()),
        }

    if method == "xlearner_rf":
        selected0 = treatment == 0
        selected1 = treatment == 1

        def forest(offset: int, n_rows: int) -> RandomForestRegressor:
            return RandomForestRegressor(
                n_estimators=150,
                min_samples_leaf=max(2, min(8, int(n_rows // 12))),
                max_features="sqrt",
                random_state=int(seed) + offset,
                n_jobs=1,
            )

        outcome0 = forest(211, int(selected0.sum())).fit(x_train[selected0], outcome[selected0])
        outcome1 = forest(307, int(selected1.sum())).fit(x_train[selected1], outcome[selected1])
        imputed0 = np.clip(outcome1.predict(x_train[selected0]), 0.0, 1.0) - outcome[selected0]
        imputed1 = outcome[selected1] - np.clip(outcome0.predict(x_train[selected1]), 0.0, 1.0)
        effect0 = forest(401, int(selected0.sum())).fit(x_train[selected0], imputed0)
        effect1 = forest(503, int(selected1.sum())).fit(x_train[selected1], imputed1)
        try:
            propensity = LogisticRegression(
                C=1.0, max_iter=1000, solver="lbfgs", random_state=int(seed)
            )
            propensity.fit(x_train, treatment)
            eval_propensity = np.clip(propensity.predict_proba(x_eval)[:, 1], 0.05, 0.95)
        except ValueError:
            eval_propensity = np.full(len(x_eval), np.mean(treatment), dtype=float)
        tau = eval_propensity * effect0.predict(x_eval) + (1.0 - eval_propensity) * effect1.predict(
            x_eval
        )
        center = (
            np.clip(outcome0.predict(x_eval), 0.0, 1.0)
            + np.clip(outcome1.predict(x_eval), 0.0, 1.0)
        ) / 2.0
        mu0 = np.clip(center - tau / 2.0, 0.0, 1.0)
        mu1 = np.clip(center + tau / 2.0, 0.0, 1.0)
        return {"model_value_t0": float(mu0.mean()), "model_value_t1": float(mu1.mean())}

    raise ValueError(f"Unknown statistical baseline: {method}")


def training_args(
    config: dict[str, Any], *, source: str, fold: int, resume: bool
) -> SimpleNamespace:
    overrides = [
        item["settings"]
        for item in config["generators"]
        if item["source"] == source and item["fold"] == fold
    ]
    if len(overrides) > 1:
        raise ValueError(f"Duplicate generator settings for {source}, fold {fold}")
    settings = config["common"] | (overrides[0] if overrides else {})
    return SimpleNamespace(**(settings | {"resume": resume}))


def model_meta(tasks: pd.DataFrame, feature_count: int) -> dict[str, Any]:
    return {
        "d_x": int(feature_count),
        "seq_len": 1,
        "a_is_discrete": True,
        "t_is_discrete": True,
        "a_vocab_size": int(tasks["task_id"].max()) + 2,
        "t_vocab_size": 2,
        "d_a": 1,
        "d_t": 1,
        "d_y": 1,
    }


def evaluate_model(
    generator: Any,
    *,
    dataset: str,
    fold: int,
    seed: int,
    tasks: pd.DataFrame,
    train_rows: pd.DataFrame,
    eval_rows: pd.DataFrame,
    x_all: np.ndarray,
    elapsed_seconds: float,
    device: torch.device,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for _, task in tasks.iterrows():
        train_task = train_rows[train_rows["task_key"].eq(task["task_key"])]
        eval_task = eval_rows[eval_rows["task_key"].eq(task["task_key"])]
        values = estimate_contextual_values(
            generator,
            eval_task,
            x_all,
            seed=int(seed),
            model_index=0,
            mc_samples=1,
            batch_size=2048,
            device=device,
        )
        records.append(
            fold_result(
                dataset=dataset,
                fold=int(fold),
                seed=int(seed),
                model=CAUSALSEQGAN_ID,
                task=task,
                train_rows=train_task,
                eval_rows=eval_task,
                values=values,
                elapsed_seconds=float(elapsed_seconds) / max(1, len(tasks)),
            )
        )
    return records


def model_knobs(model: str, args: argparse.Namespace) -> dict[str, Any]:
    knobs = dict(MODEL_KNOBS[model])
    knobs["save_checkpoints"] = False
    knobs["log_every"] = int(args.log_every)
    knobs["dropout"] = float(args.dropout)
    knobs["a_emb_dim"] = int(args.a_emb_dim)
    knobs["t_emb_dim"] = int(args.t_emb_dim)
    if model == CAUSALSEQGAN_ID:
        ea, eb, ec = [int(x) for x in args.scm_epochs]
        knobs.update(
            {
                "epochs_a": ea,
                "epochs_b": eb,
                "epochs_c": ec,
                "dynamics": str(args.scm_dynamics),
                "d_k": int(args.scm_d_k),
                "mlp_hidden": int(args.hidden),
                "y_head_hidden": int(args.scm_y_head_hidden),
                "y_head_layers": int(args.scm_y_head_layers),
                "y_head_mode": str(getattr(args, "scm_y_head_mode", "joint")),
                "k_c_ratio": float(args.scm_k_c_ratio),
                "lr_g": float(getattr(args, "scm_lr_g", 1e-3)),
                "weight_decay": float(getattr(args, "scm_weight_decay", 0.0)),
                "w_sup": float(getattr(args, "scm_w_sup", 1.0)),
                "w_adv": float(getattr(args, "scm_w_adv", 0.5)),
                "w_do": float(args.scm_w_do),
                "w_cf": float(args.scm_w_cf),
                "w_advT_causal": float(args.scm_w_advT_causal),
                "w_t_pred_spurious": float(args.scm_w_t_pred_spurious),
                "tf_ffn_hidden": max(64, int(args.hidden) * 2),
                "do_time_sampling": "fixed",
                "do_time_index": int(getattr(args, "do_time_index", 1)),
                "do_num_time_samples": 1,
                "do_min_arm_samples": int(args.do_min_arm_samples),
                "do_actions": "0,1",
                "do_horizon_train": int(getattr(args, "do_horizon_train", 0)),
                "causal_every": int(args.causal_every),
                "cf_every": int(args.cf_every),
                "ref_estimator": None,
            }
        )
    elif model == "rcgan":
        ea, eb, ec = [int(x) for x in args.gan_epochs]
        knobs.update(
            {"epochs_a": ea, "epochs_b": eb, "epochs_c": ec, "rcgan_hidden": int(args.hidden)}
        )
    elif model == "crn":
        knobs.update({"epochs": int(args.epochs), "crn_hidden": int(args.hidden)})
    elif model == "vae":
        knobs.update(
            {
                "epochs": int(args.epochs),
                "vae_enc_hidden": int(args.hidden),
                "vae_dec_hidden": int(args.hidden),
            }
        )
    elif model == "diffusion":
        knobs.update(
            {
                "epochs": int(args.epochs),
                "diff_steps": int(args.diff_steps),
                "diff_model_dim": int(args.hidden),
                "diff_backbone": str(args.diff_backbone),
            }
        )
    elif model == "timegan":
        ee, es, ej = [int(x) for x in args.timegan_epochs]
        knobs.update(
            {
                "epochs_embed": ee,
                "epochs_supervisor": es,
                "epochs_joint": ej,
                "timegan_hidden": int(args.timegan_hidden),
                "timegan_num_layers": 2,
                "timegan_z_dim": int(args.timegan_z_dim),
                "x_emb_dim": int(args.hidden),
                "max_batches_per_epoch": int(args.timegan_max_batches),
            }
        )
    return knobs


def train_or_load_model(
    model: str,
    *,
    train_batch: TrajectoryBatch,
    meta: dict[str, Any],
    seed: int,
    device: torch.device,
    run_dir: Path,
    args: argparse.Namespace,
    sample_weights: torch.Tensor | None = None,
) -> Any:
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = latest_checkpoint(run_dir, model) if args.resume else None
    if ckpt_path is None:
        _set_seed(int(seed))
        train_ds = TrajectoryDataset(
            X=train_batch.X,
            A=train_batch.A,
            T=train_batch.T,
            Y=train_batch.Y,
            mask=train_batch.mask,
        )
        batch_size = min(int(args.batch_size), max(8, len(train_ds)))
        drop_last = len(train_ds) >= batch_size * 2
        sampler = None
        if sample_weights is not None:
            weights = torch.as_tensor(sample_weights, dtype=torch.double).flatten().cpu()
            if (
                len(weights) != len(train_ds)
                or not torch.isfinite(weights).all()
                or bool((weights <= 0).any())
            ):
                raise ValueError(
                    "sample_weights must contain one finite positive value per training row"
                )
            generator = torch.Generator().manual_seed(int(seed))
            sampler = WeightedRandomSampler(
                weights,
                num_samples=len(train_ds),
                replacement=True,
                generator=generator,
            )
        train_dl = _make_dataloader(
            train_ds,
            batch_size=batch_size,
            shuffle=sampler is None,
            drop_last=drop_last,
            sampler=sampler,
        )
        knobs = model_knobs(model, args)
        print(
            f"[train] seed={seed} model={model} n_train={len(train_ds)} batch={batch_size} device={device}"
        )
        if model in {CAUSALSEQGAN_ID, "rcgan"}:
            ckpt_path = _train_adversarial_generator(
                model, train_dl=train_dl, meta=meta, device=device, out_dir=run_dir, knobs=knobs
            )
        elif model == "crn":
            ckpt_path = _train_crn(
                train_dl=train_dl, meta=meta, device=device, out_dir=run_dir, knobs=knobs
            )
        elif model == "vae":
            ckpt_path = _train_vae(
                train_dl=train_dl, meta=meta, device=device, out_dir=run_dir, knobs=knobs
            )
        elif model == "diffusion":
            ckpt_path = _train_diffusion(
                train_dl=train_dl, meta=meta, device=device, out_dir=run_dir, knobs=knobs
            )
        elif model == "timegan":
            ckpt_path = _train_timegan(
                train_dl=train_dl, meta=meta, device=device, out_dir=run_dir, knobs=knobs
            )
        else:
            raise ValueError(model)
    else:
        print(f"[resume] seed={seed} model={model} checkpoint={ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    gen_model = load_base_model_from_checkpoint(ckpt, device=device)
    gen_model.name = model
    return gen_model
