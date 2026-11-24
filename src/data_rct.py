"""Load randomised educational trials and construct binary treatment tasks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DATASET_DIRS = {
    "assistments_rct88": Path("data/rct/assistments_rct88"),
    "assistments_las2016": Path("data/rct/assistments_las2016"),
    "assistments_abtest_study2": Path("data/rct/assistments_abtest_study2"),
}


ID_COLS = {
    "dataset_name",
    "experiment_id",
    "dependent_measure",
    "independent_measure_pair",
    "condition",
    "student_id",
    "assignment_completed",
    "normalized_student_learning",
}


DATASET_LABELS = {
    "assistments_rct88": "ASSISTments RCT88/89",
    "assistments_las2016": "ASSISTments LAS2016 22 RCTs",
    "assistments_abtest_study2": "ASSISTments OSF Study2 11 A/B Tests",
}


LAS_PRETREATMENT_COLUMNS = [
    "Prior Problem Count",
    "Prior Correct Count",
    "Prior Percent Correct",
    "Class ID",
    "Class Section ID",
    "Class Grade",
    "Teacher ID",
    "Guessed Gender",
    "Birthyear",
    "school_id",
    "District ID",
    "State ID",
    "Prior Assignments Assigned",
    "Prior Assignment Count",
    "Prior Completion Count",
    "Prior Percent Completion",
    "Prior Class Percent Completion",
    "Z-Scored Mastery Speed",
    "Prior Homework Assigned",
    "Prior Homework Count",
    "Prior Homework Completion Count",
    "Prior Homework Percent Completion",
    "Prior Class Homework Percent Completion",
    "Z-Scored HW Mastery Speed",
]


STUDY2_PRETREATMENT_COLUMNS = [
    "Prior.Problem.Count",
    "Prior.Correct.Count",
    "Prior.Percent.Correct",
    "Class.ID",
    "Class.Section.ID",
    "Class.Grade",
    "Teacher.ID",
    "Guessed.Gender",
    "Birthyear",
    "Location.ID",
    "Role.Type",
    "Location.Type",
    "School.ID",
    "District.ID",
    "State.ID",
    "Prior.Assignments.Assigned",
    "Prior.Assignment.Count",
    "Prior.Completion.Count",
    "Prior.Percent.Completion",
    "Prior.Class.Percent.Completion",
    "Z.Scored.Mastery.Speed",
    "Prior.Homework.Assigned",
    "Prior.Homework.Count",
    "Prior.Homework.Completion.Count",
    "Prior.Homework.Percent.Completion",
    "Prior.Class.Homework.Percent.Completion",
    "Z.Scored.HW.Mastery.Speed",
]


def load_processed(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "processed_experiment_data.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing processed ASSISTments RCT data: {path}")
    df = pd.read_csv(path)
    df["experiment_id"] = df["experiment_id"].astype(str)
    df["dependent_measure"] = df["dependent_measure"].astype(str)
    df["independent_measure_pair"] = df["independent_measure_pair"].astype(str)
    df["condition"] = df["condition"].astype(int)
    return df


def load_las2016(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "ThisOne.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing ASSISTments LAS2016 data: {path}")
    raw = pd.read_csv(path)
    required = {"problem_set", "User ID", "InferredCondition", "complete"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"LAS2016 file is missing required columns: {missing}")

    out = pd.DataFrame(
        {
            "dataset_name": "assistments_las2016",
            "experiment_id": raw["problem_set"].astype(str),
            "independent_measure_pair": "Control vs. Experiment",
            "dependent_measure": "completion",
            "student_id": raw["User ID"].astype(str),
            "condition": raw["InferredCondition"]
            .astype(str)
            .str.strip()
            .str.upper()
            .map({"C": 0, "E": 1}),
            "assignment_completed": pd.to_numeric(raw["complete"], errors="coerce"),
        }
    )
    out["normalized_student_learning"] = out["assignment_completed"]

    if "Prior Percent Correct" in raw.columns:
        prior = pd.to_numeric(raw["Prior Percent Correct"], errors="coerce")
    elif "Prior Percent Completion" in raw.columns:
        prior = pd.to_numeric(raw["Prior Percent Completion"], errors="coerce")
    else:
        prior = pd.Series(np.nan, index=raw.index)
    out["student_prior_average_correctness"] = prior

    if "problem_set_name" in raw.columns:
        out["problem_set_name"] = raw["problem_set_name"].astype(str)

    for col in LAS_PRETREATMENT_COLUMNS:
        if col in raw.columns:
            out[col] = raw[col]

    out = out.dropna(subset=["condition", "assignment_completed"]).copy()
    out["condition"] = out["condition"].astype(int)
    out["assignment_completed"] = out["assignment_completed"].astype(float)
    out["normalized_student_learning"] = out["normalized_student_learning"].astype(float)
    return out


def las_inventory(data_dir: Path) -> pd.DataFrame:
    df = load_las2016(data_dir)
    rows = []
    for eid, g in df.groupby("experiment_id", sort=True):
        rows.append(
            {
                "experiment_id": str(eid),
                "has_condition_metadata": True,
                "has_raw_folder": True,
                "n_assignment_logs": 1,
                "n_problem_logs": 0,
                "n_student_logs": int(g["student_id"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def load_abtest_study2(data_dir: Path) -> pd.DataFrame:
    path = (
        data_dir / "Study2 11 New Experiments" / "TreatmentAssignment_Covariate_Outcome_Study2.csv"
    )
    if not path.exists():
        raise FileNotFoundError(f"Missing ASSISTments OSF Study2 TACO data: {path}")
    raw = pd.read_csv(path)
    required = {"target_sequence_id", "user_id", "complete"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"Study2 TACO file is missing required columns: {missing}")

    if "condition" in raw.columns:
        condition = pd.to_numeric(raw["condition"], errors="coerce")
    elif "Condition" in raw.columns:
        condition = (
            raw["Condition"]
            .astype(str)
            .str.lower()
            .map({"control": 0, "treatment": 1, "treatment1": 1})
        )
    else:
        raise ValueError("Study2 TACO file is missing `condition`/`Condition`.")

    out = pd.DataFrame(
        {
            "dataset_name": "assistments_abtest_study2",
            "experiment_id": raw["target_sequence_id"].astype(str),
            "independent_measure_pair": "Control vs. Treatment",
            "dependent_measure": "completion",
            "student_id": raw["user_id"].astype(str),
            "condition": condition,
            "assignment_completed": pd.to_numeric(raw["complete"], errors="coerce"),
        }
    )
    out["normalized_student_learning"] = out["assignment_completed"]
    out["student_prior_average_correctness"] = pd.to_numeric(
        raw.get("Prior.Percent.Correct"), errors="coerce"
    )

    for col in STUDY2_PRETREATMENT_COLUMNS:
        if col in raw.columns:
            out[col] = raw[col]

    out = out.dropna(subset=["condition", "assignment_completed"]).copy()
    out["condition"] = out["condition"].astype(int)
    out["assignment_completed"] = out["assignment_completed"].astype(float)
    out["normalized_student_learning"] = out["normalized_student_learning"].astype(float)
    return out


def abtest_study2_inventory(data_dir: Path) -> pd.DataFrame:
    df = load_abtest_study2(data_dir)
    rows = []
    for eid, g in df.groupby("experiment_id", sort=True):
        rows.append(
            {
                "experiment_id": str(eid),
                "has_condition_metadata": True,
                "has_raw_folder": True,
                "n_assignment_logs": 1,
                "n_problem_logs": 0,
                "n_student_logs": int(g["student_id"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def load_benchmark_data(dataset_name: str, data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if dataset_name == "assistments_rct88":
        df = load_processed(data_dir)
        df["dataset_name"] = dataset_name
        return df, raw_inventory(data_dir)
    if dataset_name == "assistments_las2016":
        return load_las2016(data_dir), las_inventory(data_dir)
    if dataset_name == "assistments_abtest_study2":
        return load_abtest_study2(data_dir), abtest_study2_inventory(data_dir)
    raise ValueError(f"Unknown dataset_name={dataset_name}")


def raw_inventory(data_dir: Path) -> pd.DataFrame:
    cond_path = data_dir / "experiment_conditions.csv"
    rows: list[dict[str, Any]] = []
    cond_ids: set[str] = set()
    if cond_path.exists():
        cond = pd.read_csv(cond_path)
        if "experiment_id" in cond.columns:
            cond_ids = set(cond["experiment_id"].astype(str))
    raw_root = data_dir / "experiment_data"
    if not raw_root.exists():
        raw_root = data_dir
    raw_ids: set[str] = set()
    if raw_root.exists():
        raw_ids = {p.name for p in raw_root.iterdir() if p.is_dir() and p.name.startswith("PSA")}
    for eid in sorted(cond_ids | raw_ids):
        p = raw_root / eid
        rows.append(
            {
                "experiment_id": eid,
                "has_condition_metadata": eid in cond_ids,
                "has_raw_folder": p.exists(),
                "n_assignment_logs": int(len(list(p.glob("*alog*.csv")))) if p.exists() else 0,
                "n_problem_logs": int(len(list(p.glob("*plog*.csv")))) if p.exists() else 0,
                "n_student_logs": int(len(list(p.glob("*slog*.csv")))) if p.exists() else 0,
            }
        )
    return pd.DataFrame(rows)


def category_from_pair(pair: str) -> str:
    pair = str(pair).replace("_No_Choice", "").replace("_Choice", "")
    left = pair.split(" vs. ")[0]
    left = left.split("_")[0]
    return left.strip()


def _binary_outcome(g: pd.DataFrame) -> np.ndarray:
    outcome = str(g["dependent_measure"].iloc[0])
    if outcome == "completion":
        return g["assignment_completed"].astype(float).to_numpy()
    return (g["normalized_student_learning"].astype(float).to_numpy() > 0.0).astype(float)


def _prior_y0(g: pd.DataFrame) -> np.ndarray:
    if "student_prior_average_correctness" not in g.columns:
        return np.full(len(g), 0.5, dtype=np.float32)
    y0 = pd.to_numeric(g["student_prior_average_correctness"], errors="coerce")
    med = float(y0.median()) if y0.notna().any() else 0.5
    y0 = y0.fillna(med).clip(0.0, 1.0)
    return y0.to_numpy(dtype=np.float32)


def build_tasks(
    df: pd.DataFrame, *, min_n: int, min_arm: int, max_tasks: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    task_rows: list[pd.DataFrame] = []
    group_cols = ["experiment_id", "independent_measure_pair", "dependent_measure"]
    for (eid, pair, outcome), g0 in df.groupby(group_cols, sort=True):
        g = g0.dropna(subset=["condition", "normalized_student_learning"]).copy()
        arms = sorted(g["condition"].astype(int).unique().tolist())
        if arms != [0, 1]:
            continue
        n0 = int((g["condition"] == 0).sum())
        n1 = int((g["condition"] == 1).sum())
        if len(g) < int(min_n) or min(n0, n1) < int(min_arm):
            continue
        y = _binary_outcome(g)
        cond = g["condition"].to_numpy()
        true_value_t0 = float(y[cond == 0].mean())
        true_value_t1 = float(y[cond == 1].mean())
        ate = float(true_value_t1 - true_value_t0)
        y_norm = g["normalized_student_learning"].astype(float).to_numpy()
        ate_norm = float(y_norm[cond == 1].mean() - y_norm[cond == 0].mean())
        task_id = len(rows)
        key = f"{eid}||{pair}||{outcome}"
        g = g.copy()
        g["task_key"] = key
        g["task_id"] = task_id
        g["y_binary"] = y.astype(np.float32)
        g["y0_prior"] = _prior_y0(g)
        task_rows.append(g)
        rows.append(
            {
                "task_key": key,
                "task_id": task_id,
                "experiment_id": str(eid),
                "independent_measure_pair": str(pair),
                "dependent_measure": str(outcome),
                "category": category_from_pair(str(pair)),
                "n": int(len(g)),
                "n0": n0,
                "n1": n1,
                "true_value_t0": true_value_t0,
                "true_value_t1": true_value_t1,
                "true_ate": ate,
                "true_ate_normalized": ate_norm,
                "y_rate": float(np.mean(y)),
            }
        )
        if max_tasks and len(rows) >= int(max_tasks):
            break
    if not rows:
        raise RuntimeError("No analyzable binary RCT tasks after filters.")
    return pd.DataFrame(rows), pd.concat(task_rows, ignore_index=True)
