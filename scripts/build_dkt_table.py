"""Build the DKT-Synth horizon table from the complete saved evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["scm_causal", "crn", "timegan", "rcgan", "vae", "diffusion"]
METRICS = ["pehe", "ate_abs_error", "policy_value_error", "policy_regret"]
STEPS = [1, 5, 10, 20, 30]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render_table(selected: pd.DataFrame) -> str:
    names = dict(zip(METRICS, ["PEHE", "ATE error", "Policy-value error", "Regret"]))
    groups = {
        length: selected[selected.steps.eq(length)].set_index("model").loc[MODELS]
        for length in STEPS
    }
    lines = [
        r"% Requires \usepackage{booktabs}",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Longitudinal counterfactual evaluation on DKT-Synth ($t_0=0$). Mean (sample SD) over three independently trained seeds; leading zeros are omitted. Lower is better. Bold marks the lowest unrounded mean for each length and metric, including ties.}",
        r"\label{tab:dkt-primary}",
        r"\fontsize{8.5}{9}\selectfont",
        r"\setlength{\tabcolsep}{2pt}",
        r"\renewcommand{\arraystretch}{0.95}",
        r"\begin{tabular*}{\linewidth}{@{}l@{\extracolsep{\fill}}rrrrr@{}}",
        r"\toprule",
        r"Method & $K=1$ & $K=5$ & $K=10$ & $K=20$ & $K=30$ \\",
        r"\midrule",
    ]
    for metric in METRICS:
        lines.append(r"\multicolumn{6}{@{}l}{\textit{" + names[metric] + r"}} \\")
        for model in MODELS:
            cells = [str(groups[STEPS[0]].loc[model, "label"])]
            for length in STEPS:
                group = groups[length]
                row = group.loc[model]
                best = bool(
                    np.isclose(
                        float(row[metric + "_mean"]),
                        float(group[metric + "_mean"].min()),
                        atol=1e-12,
                        rtol=0,
                    )
                )
                value = (
                    f"{row[metric + '_mean']:.4f}".removeprefix("0")
                    + " ("
                    + f"{row[metric + '_sd']:.4f}".removeprefix("0")
                    + ")"
                )
                cells.append(r"\textbf{" + value + "}" if best else value)
            lines.append(" & ".join(cells) + r" \\")
        if metric != METRICS[-1]:
            lines.append(r"\midrule")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular*}",
            r"\par\smallskip",
            r"\begin{minipage}{\linewidth}\fontsize{8}{9}\selectfont",
            r"$K$ outcomes correspond to $h=K-1$. Effect errors use time-averaged learner effects of changing $T_0$, with later factual actions fixed. Policy paths: Never/Always Treat, Early-on ($t<10$), Late-on ($t\geq10$); two distinct prefixes for $K\leq10$. Shared 4,000 test learners per seed; 20 Monte Carlo draws. No RCT calibration head.",
            r"\end{minipage}",
            r"\end{table*}",
        ]
    )
    return "\n".join(lines) + "\n"


def build(results: Path, destinations: list[Path]) -> Path:
    summary = pd.read_csv(results / "summary.csv")
    selected = summary[summary.t0.eq(0) & summary.steps.isin(STEPS)].copy()
    if len(selected) != 30 or selected.duplicated(["model", "steps"]).any():
        raise ValueError("Expected thirty unique method-by-horizon groups")
    if set(selected.model) != set(MODELS) or not selected.n_seeds.eq(3).all():
        raise ValueError("Every horizon must include six methods and three seeds")
    columns = [metric + ending for metric in METRICS for ending in ("_mean", "_sd")]
    if not np.isfinite(selected[columns].to_numpy()).all():
        raise ValueError("All metric means and SDs must be available")
    seed_rows = pd.read_csv(results / "metrics_by_seed.csv")
    seed_rows = seed_rows[seed_rows.t0.eq(0) & seed_rows.steps.isin(STEPS)]
    if len(seed_rows) != 90:
        raise ValueError("Expected ninety underlying method-horizon-seed rows")
    grouped = seed_rows.groupby(["model", "steps"])
    values = selected.set_index(["model", "steps"])
    for metric in METRICS:
        observed = grouped[metric].agg(["mean", "std"])
        reference = values.loc[observed.index]
        np.testing.assert_allclose(
            observed["mean"], reference[metric + "_mean"], atol=1e-12, rtol=0
        )
        np.testing.assert_allclose(observed["std"], reference[metric + "_sd"], atol=1e-12, rtol=0)

    best_methods = []
    for length in STEPS:
        group = selected[selected.steps.eq(length)].set_index("model").loc[MODELS]
        minima = {metric: float(group[metric + "_mean"].min()) for metric in METRICS}
        for model, row in group.iterrows():
            for metric in METRICS:
                best = bool(
                    np.isclose(float(row[metric + "_mean"]), minima[metric], atol=1e-12, rtol=0)
                )
                if best:
                    best_methods.append(
                        dict(
                            steps=length,
                            metric=metric,
                            model=model,
                            mean=float(row[metric + "_mean"]),
                        )
                    )

    fragments = {"table_dkt_primary.tex": render_table(selected)}
    for destination in [*destinations, results]:
        destination.mkdir(parents=True, exist_ok=True)
        for name, content in fragments.items():
            (destination / name).write_text(content, encoding="utf-8")
    selected["model_order"] = selected.model.map(
        {model: index for index, model in enumerate(MODELS)}
    )
    selected = selected.sort_values(["steps", "model_order"]).drop(columns="model_order")
    selected.to_csv(results / "table_dkt_primary_data.csv", index=False)
    manifest = dict(
        rows=30,
        horizons=STEPS,
        methods=MODELS,
        intervention_time=0,
        seeds=[42, 43, 44],
        reported_metrics=METRICS,
        formatting="mean (sample SD), original scale, four decimals without leading zeros; best unrounded mean, including ties",
        input_sha256={
            name: sha256(results / name) for name in ["summary.csv", "metrics_by_seed.csv"]
        },
        layout="one transposed table: five evaluation-length columns, four metric blocks; 8.5pt type, full linewidth, no scaling",
        table_sha256=sha256(results / "table_dkt_primary.tex"),
        best_methods=best_methods,
        parts_sha256={name: sha256(results / name) for name in fragments},
    )
    (results / "table_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return results / "table_dkt_primary.tex"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/dkt"))
    args = parser.parse_args()
    directories = [args.results / "tables"]
    print(build(args.results, directories))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
