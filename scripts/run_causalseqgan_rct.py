"""Fit, predict and evaluate CausalSeqGAN on randomised educational tasks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import CAUSALSEQGAN_ID, CAUSALSEQGAN_NAME
from src.data_rct import DATASET_DIRS, build_tasks, load_benchmark_data
from src.eval.rct_benchmark import (
    KEY,
    MODEL_ORDER,
    SEEDS,
    publish_analysis,
    validate_fold_matrix,
    write_supporting_tables,
)
from src.model.effect_head import CausalSeqGANEffectHead
from src.rct import (
    assign_participant_folds,
    evaluate_model,
    experiment_task_arm_sample_weights,
    fit_feature_transform,
    fold_result,
    make_contextual_batch,
    model_meta,
    split_audit,
    train_or_load_model,
    training_args,
)
from src.training import latest_checkpoint

DEFAULT_DIR = Path("results/rct")
DEFAULT_MODEL_DIR = Path("checkpoints/rct")
DEFAULT_CONFIG = ROOT / "configs/rct.json"


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def neural_predictions(
    tasks, rows, train, evaluation, *, source, fold, seed, config, model_dir, device, fit
):
    x_all, features, transform = fit_feature_transform(rows, train, clip=8.0)
    arguments = training_args(config, source=source, fold=fold, resume=not fit)
    run_dir = model_dir / "generator" / source / f"fold{fold}" / f"seed{seed}"
    if not fit and latest_checkpoint(run_dir, CAUSALSEQGAN_ID) is None:
        raise FileNotFoundError(f"Missing CausalSeqGAN checkpoint: {run_dir}")
    generator = train_or_load_model(
        CAUSALSEQGAN_ID,
        train_batch=make_contextual_batch(train, x_all),
        meta=model_meta(tasks, len(features)),
        seed=seed,
        device=device,
        run_dir=run_dir,
        args=arguments,
        sample_weights=experiment_task_arm_sample_weights(train),
    )
    records = evaluate_model(
        generator,
        dataset=source,
        fold=fold,
        seed=seed,
        tasks=tasks,
        train_rows=train,
        eval_rows=evaluation,
        x_all=x_all,
        elapsed_seconds=0.0,
        device=device,
    )
    transform_path = model_dir / "features" / f"{source}_fold{fold}.json"
    transform_path.parent.mkdir(parents=True, exist_ok=True)
    transform_path.write_text(json.dumps(transform, indent=2) + "\n", encoding="utf-8")
    del generator
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return pd.DataFrame(records)[KEY + ["model_value_t0", "model_value_t1"]]


def predict(args, config: dict) -> pd.DataFrame:
    priors_path = args.model_dir / "neural_priors.csv"
    priors = pd.read_csv(priors_path) if priors_path.exists() else pd.DataFrame()
    outputs, new_priors, reproduction = [], [], []
    if not args.fit and not args.rebuild_priors and priors.empty:
        raise FileNotFoundError(
            f"Missing saved neural predictions: {priors_path}. Use --fit or --rebuild-priors."
        )
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.1)
        torch.set_float32_matmul_precision("high")
    for source in args.datasets:
        raw, _ = load_benchmark_data(source, DATASET_DIRS[source])
        tasks, rows = build_tasks(raw, min_n=100, min_arm=30, max_tasks=0)
        rows = rows.reset_index(drop=True)
        assignment, metadata = assign_participant_folds(
            rows, folds=config["folds"], seed=config["fold_seed"], min_fold_arm=8, attempts=128
        )
        split_dir = args.output_dir / "splits"
        write_csv(
            split_audit(rows, assignment, dataset=source, folds=config["folds"]),
            split_dir / f"{source}.csv",
        )
        (split_dir / f"{source}.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        for fold in range(config["folds"]):
            train, evaluation = (
                rows.loc[assignment != fold].copy(),
                rows.loc[assignment == fold].copy(),
            )
            state_path = args.model_dir / "effects" / f"{source}_fold{fold}.json"
            if args.fit:
                head = CausalSeqGANEffectHead().fit(
                    tasks,
                    train,
                    source=source,
                    outer_fold=fold,
                    cache_dir=args.model_dir / "inner_statistics",
                )
                state_path.parent.mkdir(parents=True, exist_ok=True)
                head.save(state_path)
            else:
                head = CausalSeqGANEffectHead.load(state_path)
            for seed in config["seeds"]:
                if args.fit or args.rebuild_priors:
                    neural = neural_predictions(
                        tasks,
                        rows,
                        train,
                        evaluation,
                        source=source,
                        fold=fold,
                        seed=seed,
                        config=config,
                        model_dir=args.model_dir,
                        device=device,
                        fit=args.fit,
                    )
                    new_priors.append(neural)
                else:
                    neural = priors[
                        priors.dataset.eq(source) & priors.fold.eq(fold) & priors.seed.eq(seed)
                    ].copy()
                if len(neural) != len(tasks) or neural.task_key.duplicated().any():
                    raise ValueError(
                        f"Incomplete generator predictions: {source}, fold {fold}, seed {seed}"
                    )
                prediction = head.predict(
                    tasks,
                    train,
                    evaluation,
                    neural[["task_key", "model_value_t0", "model_value_t1"]],
                )
                by_task = prediction.set_index("task_key")
                for _, task in tasks.iterrows():
                    key = task["task_key"]
                    outputs.append(
                        fold_result(
                            dataset=source,
                            fold=fold,
                            seed=seed,
                            model=CAUSALSEQGAN_ID,
                            task=task,
                            train_rows=train[train.task_key.eq(key)],
                            eval_rows=evaluation[evaluation.task_key.eq(key)],
                            values=by_task.loc[key].to_dict(),
                            elapsed_seconds=0.0,
                        )
                    )
                if seed == config["seeds"][0]:
                    changed = evaluation.copy()
                    for column in [
                        "y_binary",
                        "normalized_student_learning",
                        "assignment_completed",
                    ]:
                        if column in changed:
                            changed[column] = 1 - changed[column]
                    again = head.predict(
                        tasks,
                        train,
                        changed,
                        neural[["task_key", "model_value_t0", "model_value_t1"]],
                    )
                    maximum = float(
                        abs(
                            prediction[["model_value_t0", "model_value_t1"]].to_numpy()
                            - again[["model_value_t0", "model_value_t1"]].to_numpy()
                        ).max()
                    )
                    if maximum != 0:
                        raise AssertionError("Evaluation outcomes changed model predictions")
                    reproduction.append(
                        dict(source=source, fold=fold, heldout_outcome_change_delta=maximum)
                    )
            print(f"{CAUSALSEQGAN_NAME}: {source}, fold {fold + 1}/{config['folds']}", flush=True)
    result = pd.DataFrame(outputs).sort_values(KEY).reset_index(drop=True)
    result["source_model"] = result.model
    if new_priors:
        retained = (
            priors[~priors.dataset.isin(args.datasets)] if not priors.empty else pd.DataFrame()
        )
        write_csv(
            pd.concat([retained, *new_priors], ignore_index=True).sort_values(KEY), priors_path
        )
    write_csv(result, args.output_dir / "predictions.csv")
    (args.output_dir / "prediction_checks.json").write_text(
        json.dumps(
            dict(
                model=CAUSALSEQGAN_NAME,
                rows=len(result),
                heldout_outcome_checks=reproduction,
                fitting=args.fit,
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    return result


def merge_predictions(
    matrix: pd.DataFrame, predicted: pd.DataFrame, datasets: list[str]
) -> pd.DataFrame:
    """Update CausalSeqGAN probabilities while retaining matching comparator rows."""
    replace = matrix.model.eq(CAUSALSEQGAN_ID) & matrix.dataset.isin(datasets)
    current = matrix[replace].copy()
    if current.empty:
        return pd.concat([matrix, predicted], ignore_index=True)
    columns = KEY + [
        "n_train",
        "n_eval",
        "n0_eval",
        "n1_eval",
        "fold_true_value_t0",
        "fold_true_value_t1",
    ]
    expected = current[columns].sort_values(KEY).reset_index(drop=True)
    actual = predicted[columns].sort_values(KEY).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(expected, actual, check_dtype=False, atol=1e-12, rtol=1e-12)
    except AssertionError as error:
        raise ValueError(
            "Saved comparators use different tasks or participant folds; provide a matching prediction matrix."
        ) from error
    current = current.drop(columns=["model_value_t0", "model_value_t1"]).merge(
        predicted[KEY + ["model_value_t0", "model_value_t1"]], on=KEY, validate="one_to_one"
    )
    result = pd.concat([matrix[~replace], current], ignore_index=True)
    result["source_model"] = result.model
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="By default, predict with the saved generator probabilities and fitted effect estimator.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--fit",
        action="store_true",
        help="Fit the generator and effect estimator on training participants",
    )
    mode.add_argument(
        "--summarize", action="store_true", help="Generate tables from the saved prediction matrix"
    )
    parser.add_argument(
        "--rebuild-priors",
        action="store_true",
        help="Infer generator probabilities from saved checkpoints",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Fixed model and participant-fold configuration",
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=list(DATASET_DIRS), default=list(DATASET_DIRS)
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument(
        "--fold-predictions", type=Path, default=DEFAULT_DIR / "fold_predictions.csv"
    )
    parser.add_argument(
        "--device", default="auto", help="auto, cpu, cuda, or a PyTorch device such as cuda:0"
    )
    args = parser.parse_args(argv)
    if args.summarize and args.rebuild_priors:
        parser.error("--summarize cannot be combined with --rebuild-priors")
    config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    if args.config.resolve() != DEFAULT_CONFIG.resolve():
        supplied = json.loads(args.config.read_text(encoding="utf-8"))
        config = {**config, **supplied, "common": {**config["common"], **supplied["common"]}}
    if config["folds"] != 3 or config["seeds"] != SEEDS:
        parser.error("The released RCT protocol uses three folds and model seeds 42, 43 and 44")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(3)
    matrix = (
        pd.read_csv(args.fold_predictions, dtype={"experiment_id": str})
        if args.fold_predictions.exists()
        else None
    )
    if args.summarize:
        if matrix is None:
            raise FileNotFoundError(
                f"Missing complete benchmark prediction matrix: {args.fold_predictions}"
            )
        require_full = True
    else:
        with threadpool_limits(limits=1):
            predicted = predict(args, config)
        matrix = (
            predicted if matrix is None else merge_predictions(matrix, predicted, args.datasets)
        )
        require_full = set(matrix.model) == set(MODEL_ORDER)
    matrix = validate_fold_matrix(matrix, require_full=require_full)
    write_csv(matrix, args.output_dir / "fold_predictions.csv")
    table_dirs = [args.output_dir / "tables"]
    tables = publish_analysis(
        matrix,
        args.output_dir / "analysis",
        table_dirs,
        require_full=require_full,
        fold_seed=int(config["fold_seed"]),
    )
    if require_full:
        reference_path = args.output_dir / "references/zero_effect_fold_predictions.csv"
        zero = (
            pd.read_csv(reference_path, dtype={"experiment_id": str})
            if reference_path.exists()
            else None
        )
        write_supporting_tables(matrix, tables, args.output_dir / "analysis", table_dirs, zero)
    else:
        print(
            f"Fitted-model evaluation: {matrix.model.nunique()} method(s); comparator results were not supplied."
        )
    print(
        tables["overall"][
            ["label", "ate_mae_mean", "ate_mae_sd", "ate_rmse_mean", "ate_rmse_sd"]
        ].to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
