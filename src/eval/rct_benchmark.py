"""Participant-disjoint treatment-effect evaluation and manuscript tables."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CAUSALSEQGAN_ID, CAUSALSEQGAN_NAME
from src.interventions import FAMILIES
from src.interventions import intervention_family as taxonomy

KEY = ["dataset", "fold", "seed", "task_key"]
SEEDS = [42, 43, 44]
FAMILY_ORDER = list(FAMILIES)
MODEL_ORDER = [
    CAUSALSEQGAN_ID,
    "difference_in_means",
    "ancova",
    "aipw",
    "tlearner_rf",
    "xlearner_rf",
    "rcgan",
    "crn",
    "vae",
    "diffusion",
    "timegan",
]
LABELS = dict(
    zip(
        MODEL_ORDER,
        [
            CAUSALSEQGAN_NAME,
            "Difference in means",
            "ANCOVA",
            "AIPW",
            "T-learner (RF)",
            "X-learner (RF)",
            "RCGAN",
            "CRN",
            "VAE",
            "Diffusion",
            "TimeGAN",
        ],
    )
)
TASK_METRICS = [
    "ate_mae",
    "ate_mse",
    "policy_value_error",
    "policy_regret",
    "sign_accuracy",
    "sign_accuracy_clear",
    "pooled_ate_abs_error",
]
METRICS = TASK_METRICS + ["ate_rmse"]


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def order_frame(frame: pd.DataFrame, family: bool = False) -> pd.DataFrame:
    frame = frame.copy()
    frame["_model_order"] = frame.model.map({x: i for i, x in enumerate(MODEL_ORDER)})
    columns = ["_model_order"]
    if family:
        frame["_family_order"] = frame.family.map({x: i for i, x in enumerate(FAMILY_ORDER)})
        columns.insert(0, "_family_order")
    return frame.sort_values(columns).drop(columns=columns).reset_index(drop=True)


def summarize_seed_rows(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    summary = frame.groupby(groups)[METRICS].agg(["mean", "std"]).reset_index()
    summary.columns = [
        "_".join(c).rstrip("_").replace("_std", "_sd") if isinstance(c, tuple) else c
        for c in summary.columns
    ]
    count = frame.groupby(groups, as_index=False).agg(n_seeds=("seed", "nunique"))
    return summary.merge(count, on=groups, validate="one_to_one")


def aggregate(tasks: pd.DataFrame) -> dict[str, pd.DataFrame]:
    # Overall metrics pool every task in each experiment before averaging sources.
    exp = tasks.groupby(["dataset", "experiment_id", "seed", "model", "label"], as_index=False)[
        TASK_METRICS
    ].mean()
    source = exp.groupby(["dataset", "seed", "model", "label"], as_index=False)[TASK_METRICS].mean()
    source["ate_rmse"] = np.sqrt(source.ate_mse)
    by_seed = source.groupby(["seed", "model", "label"], as_index=False)[METRICS].mean()
    by_seed["sqrt_source_mean_mse_diagnostic"] = np.sqrt(by_seed.ate_mse)
    overall = summarize_seed_rows(by_seed, ["model", "label"])
    overall["n_tasks"] = tasks[["dataset", "task_key"]].drop_duplicates().shape[0]
    overall["n_experiments"] = tasks[["dataset", "experiment_id"]].drop_duplicates().shape[0]
    overall["n_sources"] = source.dataset.nunique()
    family_exp = tasks.groupby(
        ["family", "dataset", "experiment_id", "seed", "model", "label"], as_index=False
    )[TASK_METRICS].mean()
    family_seed = family_exp.groupby(["family", "seed", "model", "label"], as_index=False)[
        TASK_METRICS
    ].mean()
    family_seed["ate_rmse"] = np.sqrt(family_seed.ate_mse)
    family_summary = summarize_seed_rows(family_seed, ["family", "model", "label"])
    reference = tasks[tasks.model.eq(tasks.model.iloc[0]) & tasks.seed.eq(SEEDS[0])]
    composition = reference.groupby("family", as_index=False).agg(
        n_tasks=("task_key", "size"),
        n_experiments=("experiment_id", "nunique"),
        n_clear_tasks=("sign_accuracy_clear", "count"),
    )
    family_summary = family_summary.merge(composition, on="family", validate="many_to_one")
    family_source = family_exp.groupby(
        ["family", "dataset", "seed", "model", "label"], as_index=False
    )[TASK_METRICS].mean()
    family_source["ate_rmse"] = np.sqrt(family_source.ate_mse)
    family_source_balanced_seed = family_source.groupby(
        ["family", "seed", "model", "label"], as_index=False
    )[METRICS].mean()
    family_source_balanced = summarize_seed_rows(
        family_source_balanced_seed, ["family", "model", "label"]
    )
    return dict(
        overall=order_frame(overall),
        global_by_seed=order_frame(by_seed),
        source_by_seed=order_frame(source),
        experiment_by_seed=order_frame(exp),
        family=order_frame(family_summary, True),
        family_by_seed=order_frame(family_seed, True),
        family_experiment_by_seed=order_frame(family_exp, True),
        family_source_by_seed=order_frame(family_source, True),
        family_source_balanced=order_frame(family_source_balanced, True),
        composition=composition,
    )


def global_family_contributions(tasks: pd.DataFrame, overall_seed: pd.DataFrame) -> pd.DataFrame:
    frame = tasks.copy()
    exp_keys = ["dataset", "experiment_id", "seed", "model"]
    frame["tasks_in_experiment"] = frame.groupby(exp_keys).task_key.transform("nunique")
    frame["experiments_in_source"] = frame.groupby(
        ["dataset", "seed", "model"]
    ).experiment_id.transform("nunique")
    source_count = float(frame.dataset.nunique())
    frame["global_task_weight"] = 1.0 / (
        source_count * frame.tasks_in_experiment * frame.experiments_in_source
    )
    frame["mae_contribution"] = frame.ate_mae * frame.global_task_weight
    frame["source_mse_contribution"] = frame.ate_mse / (
        frame.tasks_in_experiment * frame.experiments_in_source
    )
    result = frame.groupby(["dataset", "family", "seed", "model", "label"], as_index=False).agg(
        global_weight=("global_task_weight", "sum"),
        global_mae_contribution=("mae_contribution", "sum"),
        source_mse_contribution=("source_mse_contribution", "sum"),
    )
    result["source_family_weight"] = source_count * result.global_weight
    result["source_mae_contribution"] = source_count * result.global_mae_contribution
    result["global_weighted_family_mae"] = result.global_mae_contribution / result.global_weight
    source_check = result.groupby(
        ["dataset", "seed", "model"], as_index=False
    ).source_family_weight.sum()
    np.testing.assert_allclose(source_check.source_family_weight, 1.0, atol=1e-12, rtol=1e-12)
    check = result.groupby(["seed", "model"], as_index=False).agg(
        global_weight=("global_weight", "sum"), reconstructed_mae=("global_mae_contribution", "sum")
    )
    check = check.merge(
        overall_seed[["seed", "model", "ate_mae"]], on=["seed", "model"], validate="one_to_one"
    )
    np.testing.assert_allclose(check.global_weight, 1.0, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(check.reconstructed_mae, check.ate_mae, atol=1e-12, rtol=1e-12)
    rmse_check = result.groupby(
        ["dataset", "seed", "model"], as_index=False
    ).source_mse_contribution.sum()
    rmse_check["reconstructed_rmse"] = np.sqrt(rmse_check.source_mse_contribution)
    rmse_check = (
        rmse_check.groupby(["seed", "model"], as_index=False)
        .reconstructed_rmse.mean()
        .merge(
            overall_seed[["seed", "model", "ate_rmse"]], on=["seed", "model"], validate="one_to_one"
        )
    )
    np.testing.assert_allclose(
        rmse_check.reconstructed_rmse, rmse_check.ate_rmse, atol=1e-12, rtol=1e-12
    )
    return order_frame(result, True)


def latex_escape(text: str) -> str:
    return str(text).replace("&", r"\&").replace("_", r"\_").replace("%", r"\%")


def cell(
    row: pd.Series, metric: str, best: bool = False, digits: int = 4, stacked: bool = False
) -> str:
    deviation = float(row[metric + "_sd"])
    # Show small seed variation with additional decimals.
    sd_text = f"{deviation:.6f}" if 1e-12 < abs(deviation) < 5e-5 else f"{deviation:.4f}"
    if digits == 6:
        sd_text = f"{deviation:.6f}" if deviation >= 1e-12 else "0.000000"
    mean_text = f"{row[metric + '_mean']:.{digits}f}"
    if stacked:
        deviation_text = r"$\pm$ " + sd_text
        if best:
            mean_text = r"\textbf{" + mean_text + "}"
            deviation_text = r"\textbf{" + deviation_text + "}"
        return r"\shortstack[r]{" + mean_text + r"\\" + deviation_text + "}"
    value = mean_text + r" $\pm$ " + sd_text
    return r"\textbf{" + value + "}" if best else value


def write_latex(summary: pd.DataFrame, family: pd.DataFrame, output: Path) -> None:
    metrics = ["ate_mae", "ate_rmse", "policy_value_error", "policy_regret"]
    best = {m: float(summary[m + "_mean"].min()) for m in metrics}
    lines = [
        r"% Requires \usepackage{booktabs}",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Participant-disjoint randomised effect recovery and single-step decision diagnostics on 118 tasks from 76 experiments. Values are experiment-balanced and source-balanced mean $\pm$ sample SD over model seeds 42, 43, and 44, using the same fixed three-fold participant partition. Lower is better. Bold marks the best unrounded mean in each column.}",
        r"\label{tab:rct-primary}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r" & \multicolumn{2}{c}{Effect recovery} & \multicolumn{2}{c}{Decision diagnostics} \\",
        r"\cmidrule(lr){2-3}\cmidrule(l){4-5}",
        r"Method & ATE MAE & ATE RMSE & \shortstack{Arm-value\\error} & \shortstack{Policy\\regret} \\",
        r"\midrule",
    ]
    for model in MODEL_ORDER:
        if model == CAUSALSEQGAN_ID:
            lines.append(r"\multicolumn{5}{@{}l}{\textit{Proposed method}} \\")
        if model == "difference_in_means":
            lines.extend(
                [
                    r"\addlinespace[2pt]",
                    r"\multicolumn{5}{@{}l}{\textit{Direct causal estimators}} \\",
                ]
            )
        if model == "rcgan":
            lines.extend(
                [r"\addlinespace[2pt]", r"\multicolumn{5}{@{}l}{\textit{Sequence models}} \\"]
            )
        row = summary[summary.model.eq(model)].iloc[0]
        lines.append(
            latex_escape(row.label)
            + " & "
            + " & ".join(
                cell(
                    row, m, np.isclose(row[m + "_mean"], best[m], atol=1e-12, rtol=0), stacked=True
                )
                for m in metrics
            )
            + r" \\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\par\smallskip",
        r"\begin{minipage}{\linewidth}\footnotesize",
        r"Each cell shows the mean above its $\pm$ SD. RMSE is computed within each source before source averaging. Difference in means, ANCOVA, and AIPW are deterministic here and have zero seed variation. Arm-value error is the absolute calibration error for the model-selected arm; regret is the unseen-fold empirical best-arm mean minus the mean for the selected arm. The randomised arm means are sample references, not oracle potential outcomes.",
        r"\end{minipage}",
        r"\end{table}",
    ]
    (output / "table_rct_primary.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    lines = [
        r"% Requires \usepackage{booktabs,array}",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{CausalSeqGAN effect recovery and single-step decision diagnostics by intervention family, from the same 118-task prediction matrix as Table~\ref{tab:rct-primary}. Tasks are averaged within experiment and experiments are equally weighted within family; mean $\pm$ sample SD is over the three model seeds on the fixed participant partition. Family means do not use the source weights of the overall table. Lower is better. Bold marks a lowest unrounded mean among all eleven methods for that family and metric.}",
        r"\label{tab:rct-intervention-families}",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{2.8cm}rrrrrr@{}}",
        r"\toprule",
        r"Family & Tasks & Exps. & \shortstack{ATE\\MAE} & \shortstack{ATE\\RMSE} & \shortstack{Arm-value\\error} & \shortstack{Policy\\regret} \\",
        r"\midrule",
    ]
    for name in FAMILY_ORDER:
        row = family[family.family.eq(name) & family.model.eq(CAUSALSEQGAN_ID)].iloc[0]
        group = family[family.family.eq(name)]
        cells = [latex_escape(name), str(int(row.n_tasks)), str(int(row.n_experiments))]
        cells += [
            cell(
                row,
                m,
                np.isclose(row[m + "_mean"], group[m + "_mean"].min(), atol=1e-12, rtol=0),
                digits=6,
                stacked=True,
            )
            for m in metrics
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\par\smallskip",
        r"\begin{minipage}{\textwidth}\footnotesize",
        r"Each cell shows the mean above its $\pm$ SD. Some experiments contribute tasks to multiple families, so the experiment counts sum to more than 76. Arm-value error evaluates calibration of the model-selected arm; regret uses the unseen-fold empirical best arm as its sample reference. Complete comparator results are reported in Table~\ref{tab:rct-all-family-methods}.",
        r"\end{minipage}",
        r"\end{table*}",
    ]
    (output / "table_rct_six_families.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    lines = [
        r"% Requires \usepackage{booktabs,longtable,array}",
        r"\begingroup\small",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\renewcommand{\arraystretch}{1.1}",
        r"\begin{longtable}{@{}>{\raggedright\arraybackslash}p{2.7cm}>{\raggedright\arraybackslash}p{2.3cm}rrrr@{}}",
        r"\caption{All eleven methods by intervention family, using the same fold predictions as the overall results. Cells show experiment-macro means above their $\pm$ sample SD over the three model seeds on a fixed participant partition. Arm-value error measures selected-arm calibration; regret uses the unseen-fold empirical best-arm mean as its sample reference. Lower is better. Bold marks the best unrounded mean within each family and metric.}\label{tab:rct-all-family-methods}\\",
        r"\toprule Family & Method & \shortstack{ATE\\MAE} & \shortstack{ATE\\RMSE} & \shortstack{Arm-value\\error} & \shortstack{Policy\\regret} \\ \midrule\endfirsthead",
        r"\toprule Family & Method & \shortstack{ATE\\MAE} & \shortstack{ATE\\RMSE} & \shortstack{Arm-value\\error} & \shortstack{Policy\\regret} \\ \midrule\endhead",
        r"\bottomrule\endfoot",
    ]
    for name in FAMILY_ORDER:
        group = family[family.family.eq(name)]
        for index, (_, row) in enumerate(group.iterrows()):
            cells = [latex_escape(name) if index == 0 else "", latex_escape(row.label)]
            for m in metrics:
                cells.append(
                    cell(
                        row,
                        m,
                        np.isclose(row[m + "_mean"], group[m + "_mean"].min(), atol=1e-12, rtol=0),
                        digits=6,
                        stacked=True,
                    )
                )
            lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\addlinespace[3pt]")
    lines += [r"\end{longtable}", r"\endgroup"]
    (output / "table_rct_all_family_methods.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def validate_fold_matrix(folds: pd.DataFrame, *, require_full: bool = True) -> pd.DataFrame:
    """Validate prediction coverage and common randomised references."""
    required = set(
        KEY
        + [
            "model",
            "experiment_id",
            "independent_measure_pair",
            "n_train",
            "n_eval",
            "n0_eval",
            "n1_eval",
            "fold_true_value_t0",
            "fold_true_value_t1",
            "model_value_t0",
            "model_value_t1",
        ]
    )
    if missing := required - set(folds):
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    frame = folds.copy()
    frame["experiment_id"] = frame.experiment_id.astype(str)
    frame["label"] = frame.model.map(LABELS)
    if frame.empty or frame.duplicated(KEY + ["model"]).any():
        raise ValueError(
            "Prediction rows must be nonempty and unique by task, fold, seed and method"
        )
    if not set(frame.model).issubset(MODEL_ORDER) or CAUSALSEQGAN_ID not in set(frame.model):
        raise ValueError("Predictions must contain CausalSeqGAN and only supported methods")
    if require_full and set(frame.model) != set(MODEL_ORDER):
        raise ValueError("The benchmark requires CausalSeqGAN and all ten comparators")
    if require_full and len(frame) != 118 * 3 * 3 * 11:
        raise ValueError("Expected 118 tasks, three folds, three seeds and eleven methods")
    if set(frame.seed) != set(SEEDS) or set(frame.fold) != {0, 1, 2}:
        raise ValueError("Unexpected folds or model seeds")
    task_count = frame[["dataset", "task_key"]].drop_duplicates().shape[0]
    if require_full and task_count != 118:
        raise ValueError("Expected 118 distinct tasks")
    if require_full and frame[["dataset", "experiment_id"]].drop_duplicates().shape[0] != 76:
        raise ValueError("Expected 76 independent experiments")
    if len(frame) != task_count * len(SEEDS) * 3 * frame.model.nunique():
        raise ValueError("Every method must cover each task, seed and fold")
    coverage = frame.groupby(["dataset", "task_key", "seed", "model"]).fold.nunique()
    if not coverage.eq(3).all():
        raise ValueError("Every task and seed must contain all three evaluation folds")
    probabilities = frame[["model_value_t0", "model_value_t1"]].to_numpy(float)
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Invalid arm probabilities")
    columns = KEY + [
        "experiment_id",
        "independent_measure_pair",
        "n_train",
        "n_eval",
        "n0_eval",
        "n1_eval",
        "fold_true_value_t0",
        "fold_true_value_t1",
    ]
    reference = (
        frame[frame.model.eq(MODEL_ORDER[0])][columns].sort_values(KEY).reset_index(drop=True)
    )
    for _, group in frame.groupby("model"):
        actual = group[columns].sort_values(KEY).reset_index(drop=True)
        pd.testing.assert_frame_equal(reference, actual, check_dtype=False, atol=1e-12, rtol=1e-12)
    effect = frame.model_value_t1.to_numpy(float) - frame.model_value_t0.to_numpy(float)
    truth = frame.fold_true_value_t1.to_numpy(float) - frame.fold_true_value_t0.to_numpy(float)
    selected = frame.model_value_t1.to_numpy(float) >= frame.model_value_t0.to_numpy(float)
    predicted_value = np.where(selected, frame.model_value_t1, frame.model_value_t0)
    reference_value = np.where(selected, frame.fold_true_value_t1, frame.fold_true_value_t0)
    frame["fold_ate_error"] = effect - truth
    frame["fold_ate_abs_err"] = abs(effect - truth)
    frame["fold_ate_squared_err"] = (effect - truth) ** 2
    frame["fold_policy_value_abs_err"] = abs(predicted_value - reference_value)
    frame["fold_policy_regret"] = (
        np.maximum(frame.fold_true_value_t0, frame.fold_true_value_t1) - reference_value
    )
    return frame


def full_references(folds: pd.DataFrame) -> pd.DataFrame:
    records = []
    selected = folds[folds.model.eq(MODEL_ORDER[0]) & folds.seed.eq(SEEDS[0])]
    for (dataset, task), group in selected.groupby(["dataset", "task_key"], sort=False):
        n0, n1 = float(group.n0_eval.sum()), float(group.n1_eval.sum())
        p0 = float(np.average(group.fold_true_value_t0, weights=group.n0_eval))
        p1 = float(np.average(group.fold_true_value_t1, weights=group.n1_eval))
        effect = p1 - p0
        se = float(np.sqrt(p0 * (1 - p0) / n0 + p1 * (1 - p1) / n1))
        low, high = max(-1.0, effect - 1.96 * se), min(1.0, effect + 1.96 * se)
        records.append(
            dict(
                dataset=dataset,
                task_key=task,
                true_value_t0=p0,
                true_value_t1=p1,
                true_ate=effect,
                true_ate_se=se,
                true_ate_ci_low=low,
                true_ate_ci_high=high,
                effect_distinguishable=bool(low > 0 or high < 0),
            )
        )
    return pd.DataFrame(records)


def compute_task_metrics(folds: pd.DataFrame, references: pd.DataFrame) -> pd.DataFrame:
    """Compute fold-first errors and pooled task-level descriptive estimates."""
    refs = references.set_index(["dataset", "task_key"])
    records = []
    for key, group in folds.groupby(["dataset", "seed", "model", "label", "task_key"], sort=False):
        dataset, seed, model, label, task = key
        row, ref = group.iloc[0], refs.loc[(dataset, task)]
        weights = group.n_eval.to_numpy(float)
        p0, p1 = group.model_value_t0.to_numpy(float), group.model_value_t1.to_numpy(float)
        r0, r1 = group.fold_true_value_t0.to_numpy(float), group.fold_true_value_t1.to_numpy(float)
        effect, reference = p1 - p0, r1 - r0
        selected = p1 >= p0
        selected_reference = np.where(selected, r1, r0)
        pooled = float(np.average(effect, weights=weights))
        sign = float(np.sign(pooled) == np.sign(ref.true_ate))
        record = dict(
            dataset=dataset,
            seed=int(seed),
            model=model,
            label=label,
            task_key=task,
            experiment_id=str(row.experiment_id),
            family=taxonomy(row.independent_measure_pair),
            independent_measure_pair=row.independent_measure_pair,
            dependent_measure=row.get("dependent_measure", ""),
            category=row.get("category", ""),
            n_eval=int(weights.sum()),
            fold_count=int(group.fold.nunique()),
            n0_eval=int(group.n0_eval.sum()),
            n1_eval=int(group.n1_eval.sum()),
            n_train_mean=float(group.n_train.mean()),
            ate_mae=float(np.average(abs(effect - reference), weights=weights)),
            ate_mse=float(np.average((effect - reference) ** 2, weights=weights)),
            policy_value_error=float(
                np.average(abs(np.where(selected, p1, p0) - selected_reference), weights=weights)
            ),
            policy_regret=float(
                np.average(np.maximum(r0, r1) - selected_reference, weights=weights)
            ),
            sign_accuracy=sign,
            sign_accuracy_clear=sign if ref.effect_distinguishable else np.nan,
            pooled_ate_abs_error=abs(pooled - ref.true_ate),
            ate_hat=pooled,
            model_value_t0=float(np.average(p0, weights=weights)),
            model_value_t1=float(np.average(p1, weights=weights)),
        )
        record.update(ref.to_dict())
        record.update(
            ate_abs_err=record["ate_mae"],
            ate_squared_err=record["ate_mse"],
            ate_error=pooled - ref.true_ate,
            aggregate_ate_error=pooled - ref.true_ate,
            aggregate_ate_abs_err=record["pooled_ate_abs_error"],
            policy_value_abs_err=record["policy_value_error"],
            source_model=model,
            method_family="generative_model"
            if model in {CAUSALSEQGAN_ID, "rcgan", "vae", "diffusion", "crn", "timegan"}
            else "statistical_causal",
        )
        records.append(record)
    return pd.DataFrame(records)


def publish_model_analysis(
    folds: pd.DataFrame,
    references: pd.DataFrame,
    tasks: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    contributions: pd.DataFrame,
    output: Path,
    table_dirs: list[Path],
    *,
    fold_seed: int,
) -> None:
    """Write a fitted-model evaluation without implying a complete comparator benchmark."""
    for name, frame in tables.items():
        filename = {"overall": "global_metrics.csv", "family": "family_metrics.csv"}.get(
            name, name + ".csv"
        )
        write_csv(frame, output / filename)
    for name, frame in [
        ("fold_predictions", folds),
        ("task_metrics", tasks),
        ("global_family_contributions", contributions),
        ("full_task_reference", references),
    ]:
        write_csv(frame, output / f"{name}.csv")
    n_tasks = len(references)
    n_experiments = tasks[["dataset", "experiment_id"]].drop_duplicates().shape[0]
    metrics = ["ate_mae", "ate_rmse", "policy_value_error", "policy_regret"]
    header = r"Method & ATE MAE & ATE RMSE & Arm-value error & Regret \\"
    lines = [
        r"% Requires \usepackage{booktabs}",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Participant-disjoint evaluation on "
        + str(n_tasks)
        + " tasks from "
        + str(n_experiments)
        + r" experiments. Values are experiment-balanced and source-balanced mean $\pm$ sample SD over three model seeds on a fixed participant partition. Lower is better. Only the evaluated methods are shown.}",
        r"\label{tab:rct-model-summary}",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        header,
        r"\midrule",
    ]
    for _, row in tables["overall"].iterrows():
        lines.append(
            " & ".join(
                [latex_escape(row.label), *[cell(row, metric, stacked=True) for metric in metrics]]
            )
            + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    generated = {"table_rct_model_summary.tex": "\n".join(lines) + "\n"}
    lines = [
        r"% Requires \usepackage{booktabs,array}",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Intervention-family evaluation of the fitted methods. Cells are experiment-balanced means $\pm$ sample SD over three model seeds; these family means do not use the overall table's source weights. Lower is better.}",
        r"\label{tab:rct-model-families}",
        r"\small",
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{2.5cm}lrrrr@{}}",
        r"\toprule",
        "Family & " + header,
        r"\midrule",
    ]
    for _, row in tables["family"].iterrows():
        lines.append(
            " & ".join(
                [
                    latex_escape(row.family),
                    latex_escape(row.label),
                    *[cell(row, metric, stacked=True) for metric in metrics],
                ]
            )
            + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    generated["table_rct_model_families.tex"] = "\n".join(lines) + "\n"
    for directory in [output, *table_dirs]:
        directory.mkdir(parents=True, exist_ok=True)
        for name, content in generated.items():
            (directory / name).write_text(content, encoding="utf-8")
    metadata = dict(
        model="CausalSeqGAN",
        complete_comparator_benchmark=False,
        evaluated_methods=list(tables["overall"].label),
        tasks=n_tasks,
        experiments=n_experiments,
        methods=int(folds.model.nunique()),
        sources=int(folds.dataset.nunique()),
        folds=3,
        seeds=SEEDS,
        participant_fold_seed=fold_seed,
        fold_rows=len(folds),
        global_mae="fold-weighted task errors, experiment mean, source mean, equal-source mean",
        global_rmse="source sqrt(experiment mean MSE), equal-source mean",
        family="experiment macro within family",
        global_reconstruction_verified=True,
    )
    (output / "summary_manifest.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def publish_analysis(
    folds: pd.DataFrame,
    output: Path,
    table_dirs: list[Path] | None = None,
    *,
    require_full: bool = True,
    fold_seed: int = 2026110201,
) -> dict[str, pd.DataFrame]:
    """Publish effect-recovery summaries and tables for the evaluated methods."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    folds = validate_fold_matrix(folds, require_full=require_full)
    references = full_references(folds)
    tasks = compute_task_metrics(folds, references)
    tables = aggregate(tasks)
    contributions = global_family_contributions(tasks, tables["global_by_seed"])
    if not require_full:
        publish_model_analysis(
            folds,
            references,
            tasks,
            tables,
            contributions,
            output,
            table_dirs or [],
            fold_seed=fold_seed,
        )
        return tables
    comparisons = []
    for name in FAMILY_ORDER:
        group = tables["family"][tables["family"].family.eq(name)]
        actual = group[group.model.eq(CAUSALSEQGAN_ID)].iloc[0]
        best = group[~group.model.eq(CAUSALSEQGAN_ID)].sort_values("ate_mae_mean").iloc[0]
        delta = float(actual.ate_mae_mean - best.ate_mae_mean)
        comparisons.append(
            dict(
                family=name,
                n_tasks=int(actual.n_tasks),
                n_experiments=int(actual.n_experiments),
                candidate_ate_mae=float(actual.ate_mae_mean),
                candidate_ate_mae_sd=float(actual.ate_mae_sd),
                candidate_ate_rmse=float(actual.ate_rmse_mean),
                candidate_ate_rmse_sd=float(actual.ate_rmse_sd),
                best_comparator=best.model,
                best_comparator_label=best.label,
                best_comparator_ate_mae=float(best.ate_mae_mean),
                mae_delta=delta,
                candidate_strict_best=bool(delta < -1e-12),
                candidate_best_or_tied=bool(delta <= 1e-12),
            )
        )
    comparisons = pd.DataFrame(comparisons)
    for name, frame in tables.items():
        filename = {
            "overall": "all_methods_global_metrics.csv",
            "family": "all_methods_family_metrics.csv",
        }.get(name, name + ".csv")
        write_csv(frame, output / filename)
    for name, frame in [
        ("all_methods_fold_predictions", folds),
        ("all_methods_task_metrics", tasks),
        ("global_reconstruction_by_family_source_seed", contributions),
        ("candidate_vs_family_comparators", comparisons),
        ("full_task_reference", references),
    ]:
        write_csv(frame, output / f"{name}.csv")
    for alias, frame in [
        ("rct_crossfit_summary", tables["overall"]),
        ("rct_crossfit_all_methods", tasks),
        ("rct_crossfit_metrics_by_seed", tables["global_by_seed"]),
        ("rct_crossfit_by_experiment_seed", tables["experiment_by_seed"]),
    ]:
        write_csv(frame, output / f"{alias}.csv")
    write_latex(tables["overall"], tables["family"], output)
    for directory in table_dirs or []:
        directory.mkdir(parents=True, exist_ok=True)
        for name in [
            "table_rct_primary.tex",
            "table_rct_six_families.tex",
            "table_rct_all_family_methods.tex",
        ]:
            (directory / name).write_bytes((output / name).read_bytes())
        taxonomy_table = (output / "table_rct_six_families.tex").read_text(encoding="utf-8")
        taxonomy_table = taxonomy_table.replace(
            r"\label{tab:rct-intervention-families}", r"\label{tab:s-rct-taxonomy}"
        )
        taxonomy_table = taxonomy_table.replace(
            r"as Table~\ref{tab:rct-primary}", "as the overall randomised benchmark"
        )
        (directory / "table_s7_rct_taxonomy.tex").write_text(taxonomy_table, encoding="utf-8")
    metadata = dict(
        model="CausalSeqGAN",
        tasks=len(references),
        experiments=tasks[["dataset", "experiment_id"]].drop_duplicates().shape[0],
        methods=folds.model.nunique(),
        sources=folds.dataset.nunique(),
        folds=3,
        seeds=SEEDS,
        participant_fold_seed=fold_seed,
        fold_rows=len(folds),
        task_seed_rows=len(tasks),
        family_rows=len(tables["family"]),
        global_mae="fold-weighted task errors, experiment mean, source mean, equal-source mean",
        global_rmse="source sqrt(experiment mean MSE), equal-source mean",
        family="experiment macro within family",
        global_reconstruction_verified=True,
    )
    (output / "summary_manifest.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return tables


def write_supporting_tables(
    folds: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    output: Path,
    table_dirs: list[Path],
    zero_folds: pd.DataFrame | None = None,
) -> None:
    selected = folds[folds.model.eq(CAUSALSEQGAN_ID) & folds.seed.eq(SEEDS[0])]
    labels = {
        "assistments_abtest_study2": "Study2",
        "assistments_las2016": "LAS2016",
        "assistments_rct88": "RCT88/89",
    }
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Randomised benchmark after uniform sample-size filters. Evaluation counts show the range over three participant-disjoint folds.}",
        r"\label{tab:rct-composition}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"Source & Tasks & Experiments & Rows & Fold range \\",
        r"\midrule",
    ]
    composition = []
    for source, label in labels.items():
        group = selected[selected.dataset.eq(source)]
        counts = group.groupby("fold").n_eval.sum()
        row = dict(
            dataset=source,
            label=label,
            tasks=group.task_key.nunique(),
            experiments=group.experiment_id.nunique(),
            rows=int(group.n_eval.sum()),
            fold_min=int(counts.min()),
            fold_max=int(counts.max()),
        )
        composition.append(row)
        lines.append(
            f"{label} & {row['tasks']} & {row['experiments']} & {row['rows']:,} & {row['fold_min']:,} to {row['fold_max']:,} "
            + r"\\"
        )
    total = selected.groupby("fold").n_eval.sum()
    lines += [
        r"\midrule",
        f"Total & 118 & 76 & {int(selected.n_eval.sum()):,} & {int(total.min()):,} to {int(total.max()):,} "
        + r"\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    outputs = {"table_rct_composition.tex": "\n".join(lines) + "\n"}
    write_csv(pd.DataFrame(composition), output / "rct_crossfit_composition.csv")
    if zero_folds is not None:
        references = full_references(folds)
        zero_tasks = compute_task_metrics(zero_folds, references)
        zero = aggregate(zero_tasks)["overall"].iloc[0]
        write_csv(aggregate(zero_tasks)["overall"], output / "zero_effect_global_diagnostics.csv")
        summary = tables["overall"]
        clear_tasks = int(references.effect_distinguishable.sum())
        lines = [
            r"\begin{table*}[t]",
            r"\centering",
            r"\caption{Secondary diagnostics for the single-step randomised benchmark. Clear-effect sign uses the "
            + str(clear_tasks)
            + r" tasks whose randomised 95\% interval excludes zero. Skill is one minus the error divided by the corresponding zero-effect error.}",
            r"\label{tab:s-rct-crossfit}",
            r"\footnotesize",
            r"\setlength{\tabcolsep}{3pt}",
            r"\begin{tabular}{@{}lrrrrr@{}}",
            r"\toprule",
            r"Method & Value error$\downarrow$ & Regret$\downarrow$ & Clear sign$\uparrow$ & MAE skill$\uparrow$ & RMSE skill$\uparrow$ \\",
            r"\midrule",
        ]
        for row in [zero, *[summary[summary.model.eq(model)].iloc[0] for model in MODEL_ORDER]]:
            mae_skill = 1.0 - float(row.ate_mae_mean) / float(zero.ate_mae_mean)
            rmse_skill = 1.0 - float(row.ate_rmse_mean) / float(zero.ate_rmse_mean)
            lines.append(
                " & ".join(
                    [
                        latex_escape(row.label),
                        cell(row, "policy_value_error"),
                        cell(row, "policy_regret"),
                        cell(row, "sign_accuracy_clear"),
                        f"{mae_skill:+.3f}",
                        f"{rmse_skill:+.3f}",
                    ]
                )
                + r" \\"
            )
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
        outputs["table_s3_rct_crossfit.tex"] = "\n".join(lines) + "\n"
    for name, content in outputs.items():
        (output / name).write_text(content, encoding="utf-8")
        for directory in table_dirs:
            (directory / name).write_text(content, encoding="utf-8")
