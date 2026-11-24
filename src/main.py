from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, TextIO

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    _repo_root_for_imports = Path(__file__).resolve().parents[1]
    if str(_repo_root_for_imports) not in sys.path:
        sys.path.insert(0, str(_repo_root_for_imports))

from src.baselines import load_base_model_from_checkpoint
from src.causal_estimators import IPTWMSM, GFormula, PrecomputedOracleEstimator
from src.config import (
    ACTIVE_DATASETS,
    CAUSALSEQGAN_ID,
    CAUSALSEQGAN_NAME,
    COMMON_KNOBS,
    DATASET_PATHS,
    DATASET_SPECIFIC_KNOBS,
    MODEL_KNOBS,
)
from src.data import (
    NPZSequenceDataset,
    TrajectoryBatch,
    TrajectoryDataset,
    compute_lengths,
    make_trajectory_dataloader,
    move_batch,
)
from src.eval.dr_eval import compute_dr_policy_values
from src.eval.oracle_metrics import compute_oracle_metrics
from src.eval.tstr_trts import run_tstr_trts
from src.model.causalseqgan import CausalSeqGAN
from src.privacy.mia_shadow import mia_to_dataframe, run_membership_inference
from src.privacy.nn_distance import compute_nn_distance
from src.training import (
    _device_from_arg,
    _make_dataloader,
    _set_seed,
    _split_indices,
    _train_adversarial_generator,
    _train_crn,
    _train_diffusion,
    _train_timegan,
    _train_vae,
    latest_checkpoint,
)
from src.utils.manifest import add_artifact, write_manifest
from src.vis import run_visualization


def _resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _dataset_meta(ds: Any) -> dict[str, Any]:
    return {
        "d_x": int(ds.d_x),
        "seq_len": int(ds.seq_len),
        "a_is_discrete": bool(ds.a_is_discrete),
        "t_is_discrete": bool(ds.t_is_discrete),
        "a_vocab_size": ds.a_vocab_size,
        "t_vocab_size": ds.t_vocab_size,
        "d_a": int(ds.d_a),
        "d_t": int(ds.d_t),
        "d_y": int(ds.d_y),
    }


def _batch_from_dataset(ds: Any) -> TrajectoryBatch:
    X = torch.as_tensor(ds.X).float()
    A = torch.as_tensor(ds.A)
    T = torch.as_tensor(ds.T).long()
    Y = torch.as_tensor(ds.Y).float()
    M = torch.as_tensor(ds.M).float()
    ids = torch.arange(X.shape[0], dtype=torch.long)
    return TrajectoryBatch(X=X, A=A, T=T, Y=Y, mask=M, lengths=compute_lengths(M), ids=ids)


def _subset_batch(batch: TrajectoryBatch, indices: list[int]) -> TrajectoryBatch:
    if not indices:
        return TrajectoryBatch(
            X=batch.X[:0],
            A=batch.A[:0],
            T=batch.T[:0],
            Y=batch.Y[:0],
            mask=batch.mask[:0],
            lengths=batch.lengths[:0],
            ids=batch.ids[:0] if batch.ids is not None else None,
        )
    idx = torch.as_tensor(indices, dtype=torch.long)
    return TrajectoryBatch(
        X=batch.X[idx],
        A=batch.A[idx],
        T=batch.T[idx],
        Y=batch.Y[idx],
        mask=batch.mask[idx],
        lengths=batch.lengths[idx],
        ids=batch.ids[idx] if batch.ids is not None else None,
    )


def _parse_int_list(text: str, *, default: list[int]) -> list[int]:
    if text is None:
        return list(default)
    items = [s.strip() for s in str(text).split(",") if s.strip()]
    return [int(x) for x in items] if items else list(default)


def _parse_str_list(text: str, *, default: list[str]) -> list[str]:
    if text is None:
        return list(default)
    items = [s.strip() for s in str(text).split(",") if s.strip()]
    return items if items else list(default)


def _resolve_horizon(horizon_arg: Optional[int], *, seq_len: int) -> int:
    seq_len = int(seq_len)
    if seq_len <= 0:
        return 0
    if horizon_arg is None:
        horizon = seq_len - 1
    else:
        horizon = int(horizon_arg)
    if horizon < 0:
        horizon = seq_len - 1
    return max(0, min(seq_len - 1, horizon))


def _build_subgroups(batch: TrajectoryBatch, names: list[str]) -> list[dict]:
    subgroups = []
    x0 = None
    if batch.X.shape[1] > 0:
        x0 = batch.X[:, 0].detach().cpu().numpy()
    else:
        y_mean = (batch.Y * batch.mask).sum(dim=1) / batch.mask.sum(dim=1).clamp(min=1.0)
        x0 = y_mean.detach().cpu().numpy()

    median = float(np.median(x0))
    for name in names:
        if name == "all":
            subgroups.append({"name": "all"})
        elif name == "low":
            subgroups.append({"name": "low", "mask": x0 <= median})
        elif name == "high":
            subgroups.append({"name": "high", "mask": x0 > median})
    return subgroups


def _write_vis_summary_models(report_f: TextIO, *, vis_out_dir: Path) -> None:
    """Append the visualizer's per-model summary metrics to the main report."""

    summary_models_csv = Path(vis_out_dir) / "summary_models.csv"
    if not summary_models_csv.exists():
        report_f.write("  vis_summary_models_csv: (missing)\n")
        return

    def _format_path(value: str) -> str:
        if not value:
            return ""
        try:
            p = Path(value)
            if not p.is_absolute():
                p = (Path(vis_out_dir) / p).resolve()
            else:
                p = p.resolve()
            try:
                return p.relative_to(Path.cwd()).as_posix()
            except Exception:
                return p.as_posix()
        except Exception:
            return str(value)

    report_f.write(f"  vis_summary_models_csv: {summary_models_csv.as_posix()}\n")
    report_f.write("  vis_summary_models:\n")

    try:
        with summary_models_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                report_f.write("    (empty)\n\n")
                return

            path_fields = {k for k in reader.fieldnames if k.endswith("_png") or k.endswith("_csv")}
            for row in reader:
                model = (row.get("model") or "").strip() or "UNKNOWN"
                report_f.write(f"    [model={model}]\n")
                for key in reader.fieldnames:
                    if key == "model":
                        continue
                    value = (row.get(key) or "").strip()
                    if key in path_fields:
                        value = _format_path(value)
                    report_f.write(f"      {key}: {value}\n")
                report_f.write("\n")
    except Exception as e:
        report_f.write(f"  vis_summary_models_error: {e}\n\n")


def write_results_summary(
    metrics_tables: dict[str, pd.DataFrame],
    out_txt: str,
) -> None:
    out_path = Path(out_txt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    id_candidates = [
        "model",
        "dataset",
        "ref_estimator",
        "predictor",
        "setting",
        "policy",
        "metric",
        "publish_mode",
    ]
    with out_path.open("w", encoding="utf-8") as f:
        for name, df in metrics_tables.items():
            f.write(f"{name}\n")
            if df is None or df.empty:
                f.write("(empty)\n\n")
                continue
            id_cols = [c for c in id_candidates if c in df.columns]
            num_cols = list(df.select_dtypes(include=[np.number]).columns)
            num_cols = [c for c in num_cols if c not in id_cols]
            if not id_cols and not num_cols:
                f.write("(empty)\n\n")
                continue
            out_df = df[id_cols + num_cols].copy()
            sort_cols = [
                c for c in ["model", "predictor", "setting", "policy"] if c in out_df.columns
            ]
            if sort_cols:
                out_df = out_df.sort_values(sort_cols)
            f.write(out_df.to_string(index=False))
            f.write("\n\n")

        oracle_ite = metrics_tables.get("oracle_ite_metrics")
        oracle_policy = metrics_tables.get("oracle_policy_metrics")
        if oracle_ite is not None and not oracle_ite.empty and "supported" in oracle_ite.columns:
            supported_col = oracle_ite["supported"].fillna(False).astype(bool)
            unsupported = oracle_ite[~supported_col]
            if not unsupported.empty:
                ratio = float(len(unsupported) / max(1, len(oracle_ite)))
                datasets = sorted({str(x) for x in unsupported["dataset"].dropna().tolist()})
                f.write("oracle_availability\n")
                f.write(f"unsupported_ratio: {ratio:.3f}\n")
                f.write(
                    f"datasets_without_oracle: {', '.join(datasets) if datasets else '(none)'}\n\n"
                )

        if oracle_ite is not None and not oracle_ite.empty:
            if "dataset" in oracle_ite.columns:
                ite_irt = oracle_ite[oracle_ite["dataset"] == "irt_synth"]
            else:
                ite_irt = oracle_ite.iloc[0:0]
            if "supported" in ite_irt.columns:
                ite_irt = ite_irt[ite_irt["supported"] == True]
            pehe_vals = ite_irt.get("pehe") if "pehe" in ite_irt.columns else None
            ate_rmse_vals = ite_irt.get("ate_rmse") if "ate_rmse" in ite_irt.columns else None

            def _stat_line(values: Optional[pd.Series], label: str) -> str:
                if values is None:
                    return f"{label}: mean=nan median=nan p25=nan p75=nan"
                arr = values.to_numpy(dtype=np.float64)
                arr = arr[np.isfinite(arr)]
                if arr.size == 0:
                    return f"{label}: mean=nan median=nan p25=nan p75=nan"
                return (
                    f"{label}: mean={float(np.mean(arr)):.6f} "
                    f"median={float(np.median(arr)):.6f} "
                    f"p25={float(np.quantile(arr, 0.25)):.6f} "
                    f"p75={float(np.quantile(arr, 0.75)):.6f}"
                )

            f.write("oracle_irt_synth_summary\n")
            f.write(_stat_line(pehe_vals, "pehe"))
            f.write("\n")
            f.write(_stat_line(ate_rmse_vals, "ate_rmse"))
            f.write("\n")

            regret_vals = None
            if (
                oracle_policy is not None
                and not oracle_policy.empty
                and "dataset" in oracle_policy.columns
            ):
                pol_irt = oracle_policy[oracle_policy["dataset"] == "irt_synth"]
                if "supported" in pol_irt.columns:
                    pol_irt = pol_irt[pol_irt["supported"] == True]
                if "regret_oracle" in pol_irt.columns and "model" in pol_irt.columns:
                    reg = pol_irt[["model", "regret_oracle"]].dropna()
                    reg = reg[np.isfinite(reg["regret_oracle"].to_numpy(dtype=np.float64))]
                    if not reg.empty:
                        regret_vals = reg.drop_duplicates()["regret_oracle"]
            f.write(_stat_line(regret_vals, "policy_regret"))
            f.write("\n\n")


def _compute_factual_auc(
    gen_model: Any,
    *,
    data: TrajectoryBatch,
    batch_size: int,
) -> tuple[float, int, str]:
    model = getattr(gen_model, "model", None)
    model_name = str(getattr(gen_model, "name", ""))
    if model is None or not hasattr(model, "parameters"):
        return float("nan"), 0, "no_model"

    device = next(model.parameters()).device
    ds = TrajectoryDataset(
        X=data.X.detach().cpu(),
        A=data.A.detach().cpu(),
        T=data.T.detach().cpu(),
        Y=data.Y.detach().cpu(),
        mask=data.mask.detach().cpu(),
    )
    dl = make_trajectory_dataloader(ds, batch_size=int(batch_size), shuffle=False, drop_last=False)
    y_true_all: list[np.ndarray] = []
    y_pred_all: list[np.ndarray] = []

    with torch.no_grad():
        for batch in dl:
            batch = move_batch(batch, device)
            X, A, T, Y, M = batch.X, batch.A, batch.T, batch.Y, batch.mask

            if isinstance(model, CausalSeqGAN):
                out = model.teacher_forcing(  # type: ignore[attr-defined]
                    x=X, a=A, t=T, y=Y, mask=M, eps=None, eps_mode="zero"
                )
                y_prob = torch.sigmoid(out["y_logits"])
            elif model_name == "rcgan":
                d_eps = int(getattr(getattr(model, "cfg", None), "d_eps", 16))
                eps = torch.zeros(A.shape[0], A.shape[1], d_eps, device=device)
                out = model.teacher_forcing(  # type: ignore[attr-defined]
                    x=X, a=A, t=T, y=Y, mask=M, eps=eps, stochastic_y=False
                )
                y_prob = torch.sigmoid(out["y_logits"])
            elif model_name == "crn":
                out = model.forward(x=X, a=A, t=T, y=Y, mask=M)  # type: ignore[call-arg]
                y_prob = torch.sigmoid(out["y_logits"])
            elif model_name == "vae":
                mu, _logvar = model.encode(x=X, a=A, t=T, y=Y, mask=M)  # type: ignore[attr-defined]
                out = model.decode(  # type: ignore[attr-defined]
                    x=X, a=A, t=T, mask=M, z=mu, y=Y, teacher_forcing=True, stochastic_y=False
                )
                y_prob = torch.sigmoid(out["y_logits"])
            elif model_name == "timegan":
                z_dim = int(getattr(getattr(model, "cfg", None), "z_dim", 16))
                z = torch.zeros(A.shape[0], A.shape[1], z_dim, device=device)
                out = model.teacher_forcing(  # type: ignore[attr-defined]
                    x=X, a=A, t=T, y=None, mask=M, z=z, stochastic_y=False
                )
                y_prob = torch.sigmoid(out["y_logits"])
            else:
                return float("nan"), 0, f"unsupported model={model_name}"

            if y_prob.ndim == 3:
                y_prob = y_prob.squeeze(-1)
            if Y.ndim == 3:
                Y = Y.squeeze(-1)

            valid = M > 0.5
            if not valid.any():
                continue
            y_true_all.append(Y[valid].detach().cpu().numpy())
            y_pred_all.append(y_prob[valid].detach().cpu().numpy())

    if not y_true_all:
        return float("nan"), 0, "empty"
    y_true = np.concatenate(y_true_all, axis=0)
    y_pred = np.concatenate(y_pred_all, axis=0)
    if np.unique(y_true).size < 2:
        return float("nan"), int(y_true.size), "single_class"
    return float(roc_auc_score(y_true, y_pred)), int(y_true.size), ""


def _compute_t_leakage_auc(
    gen_model: Any,
    *,
    train_batch: TrajectoryBatch,
    test_batch: TrajectoryBatch,
    max_points: int = 20000,
    epochs: int = 3,
) -> float:
    model = getattr(gen_model, "model", None)
    if not isinstance(model, CausalSeqGAN):
        return float("nan")
    if not bool(getattr(model.cfg, "t_is_discrete", True)):
        return float("nan")
    if train_batch.T.ndim != 2 or test_batch.T.ndim != 2:
        return float("nan")
    if int(getattr(model, "d_k_c", 0)) <= 0:
        return float("nan")

    def _extract_kc(batch: TrajectoryBatch) -> tuple[np.ndarray, np.ndarray]:
        device = next(model.parameters()).device
        X = batch.X.to(device)
        A = batch.A.to(device)
        T = batch.T.to(device)
        Y = batch.Y.to(device)
        M = batch.mask.to(device)
        with torch.no_grad():
            tf = model.teacher_forcing(x=X, a=A, t=T, y=Y, mask=M, eps=None, eps_mode="zero")
        k_c = tf.get("k_c")
        if k_c is None:
            k_c, _ = model.split_k(tf["k"])
        k_c = k_c[:, :-1, :]
        valid = M > 0.5
        if not valid.any():
            return np.zeros((0, k_c.shape[-1]), dtype=np.float32), np.zeros((0,), dtype=np.int64)
        k_flat = k_c[valid].detach().cpu().numpy().astype(np.float32)
        t_flat = T[valid].detach().cpu().numpy().astype(np.int64).reshape(-1)
        return k_flat, t_flat

    k_train, t_train = _extract_kc(train_batch)
    k_test, t_test = _extract_kc(test_batch)
    if k_train.size == 0 or k_test.size == 0:
        return float("nan")

    classes = np.unique(np.concatenate([t_train, t_test], axis=0))
    if classes.size < 2:
        return float("nan")
    class_map = {int(c): i for i, c in enumerate(classes)}
    t_train_idx = np.vectorize(lambda x: class_map[int(x)])(t_train)
    t_test_idx = np.vectorize(lambda x: class_map[int(x)])(t_test)
    num_classes = int(classes.size)

    rng = np.random.default_rng(0)
    if k_train.shape[0] > int(max_points):
        sel = rng.choice(k_train.shape[0], size=int(max_points), replace=False)
        k_train = k_train[sel]
        t_train_idx = t_train_idx[sel]

    device = next(model.parameters()).device
    x_train = torch.as_tensor(k_train, device=device, dtype=torch.float32)
    y_train = torch.as_tensor(t_train_idx, device=device, dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(x_train, y_train)
    dl = DataLoader(dataset, batch_size=1024, shuffle=True)

    clf = torch.nn.Linear(x_train.shape[1], num_classes).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-2)
    clf.train()
    for _ in range(max(1, int(epochs))):
        for xb, yb in dl:
            logits = clf(xb)
            loss = torch.nn.functional.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    x_test = torch.as_tensor(k_test, device=device, dtype=torch.float32)
    with torch.no_grad():
        logits = clf(x_test)
        prob = torch.softmax(logits, dim=-1).detach().cpu().numpy()

    if num_classes == 2:
        return _roc_auc_score(t_test_idx, prob[:, 1])
    aucs = []
    for cls in range(num_classes):
        y_true = (t_test_idx == cls).astype(np.int64)
        auc = _roc_auc_score(y_true, prob[:, cls])
        if np.isfinite(auc):
            aucs.append(auc)
    if not aucs:
        return float("nan")
    return float(np.mean(aucs))


def _maybe_plot_cf_consistency_multi(
    *,
    ckpt_paths: dict[str, Path],
    model_order: list[str],
    test_batch: TrajectoryBatch,
    dataset: str,
    t0_list: list[int],
    out_dir: Path,
    seed: int,
    actions: list[int],
    plot_horizon: int,
    device: torch.device,
) -> Optional[Path]:
    if dataset not in {"assist09", "oulad", "statics"}:
        return None

    try:
        import matplotlib

        matplotlib.use("Agg")
        from scripts import eval_cf_consistency as cf_vis
    except Exception as e:
        print(f"[cf_consistency] import failed: {e}")
        return None

    if not ckpt_paths:
        return None

    models: list[tuple[str, object, str]] = []
    for model_key in model_order:
        ckpt_path = ckpt_paths.get(model_key)
        if ckpt_path is None:
            continue
        try:
            ckpt = torch.load(ckpt_path, map_location=device)
            if model_key == CAUSALSEQGAN_ID:
                model, _ = cf_vis.load_rollout_model_from_checkpoint(ckpt, device=device)
                models.append((model_key, model, "scm_raw"))
            else:
                base_model = load_base_model_from_checkpoint(ckpt, device=device)
                models.append((model_key, base_model, "wrapped"))
        except Exception as e:
            print(f"[cf_consistency] skip {model_key}: {e}")

    if not models:
        return None

    seq_len = int(test_batch.T.shape[1])
    if seq_len <= 1:
        return None

    action_a = int(actions[0]) if actions else 0
    action_b = int(actions[1]) if len(actions) > 1 else 1
    t_intervention = int(t0_list[0]) if t0_list else min(10, seq_len - 1)
    t_intervention = max(0, min(int(t_intervention), seq_len - 1))
    plot_horizon = max(1, min(int(plot_horizon), seq_len - 1))

    _, ref_model, ref_mode = next(
        (item for item in models if item[0] == CAUSALSEQGAN_ID), models[0]
    )
    if ref_mode != "scm_raw":
        student_idx = 0
    else:
        student_idx = cf_vis.select_student_idx_for_model(
            ref_model,
            test_batch,
            t_intervention=t_intervention,
            action_a=action_a,
            action_b=action_b,
            plot_horizon=plot_horizon,
            seed=int(seed),
            select="max_delta_k",
            causal_only=True,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"cf_bifurcation_{dataset}_all.png"
    out_data = out_dir / f"cf_bifurcation_{dataset}_all.npz"
    try:
        cf_vis.plot_bifurcation_multi(
            models=models,
            batch=test_batch,
            student_idx=int(student_idx),
            t_intervention=t_intervention,
            action_a=action_a,
            action_b=action_b,
            plot_horizon=plot_horizon,
            seed=int(seed),
            out_path=str(out_path),
            out_data_path=str(out_data),
        )
    except Exception as e:
        print(f"[cf_consistency] multi-plot failed: {e}")
        return None
    return out_path


def _roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(np.int64)
    y_score = np.asarray(y_score).astype(np.float64)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score, kind="mergesort")
    scores = y_score[order]
    labels = y_true[order]

    ranks = np.empty_like(scores, dtype=np.float64)
    i = 0
    rank = 1.0
    while i < scores.shape[0]:
        j = i + 1
        while j < scores.shape[0] and scores[j] == scores[i]:
            j += 1
        avg_rank = 0.5 * (rank + (rank + (j - i) - 1.0))
        ranks[i:j] = avg_rank
        rank += float(j - i)
        i = j

    sum_ranks_pos = float(ranks[labels == 1].sum())
    auc = (sum_ranks_pos - (n_pos * (n_pos + 1) / 2.0)) / float(n_pos * n_neg)
    return float(auc)


def main(argv: Optional[list[str]] = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "train":
        arguments.pop(0)
    if arguments and arguments[0] in {"rct", "--rct"}:
        from scripts.run_causalseqgan_rct import main as run_rct

        return run_rct(arguments[1:])
    if arguments and arguments[0] == "multistep":
        from scripts.evaluate_multistep import main as evaluate

        return evaluate(arguments[1:]) or 0
    p = argparse.ArgumentParser(
        description="CausalSeqGAN training and evaluation",
        epilog="Commands: train (default), rct, multistep. Use COMMAND --help for options.",
    )
    p.add_argument(
        "--data-npz", type=Path, help="Use a custom trajectory NPZ instead of a named dataset"
    )
    p.add_argument("--datasets", type=str, default=None, help="Comma-separated dataset keys")
    p.add_argument(
        "--models", type=str, default=None, help="Comma-separated models (default: CausalSeqGAN)"
    )
    p.add_argument(
        "--baseline", type=str, default=None, help="Alias for --models with a single key"
    )
    p.add_argument(
        "--out_root",
        type=str,
        default="checkpoints/trajectories",
        help="Root directory for checkpoints",
    )
    p.add_argument(
        "--results_dir", type=str, default="results", help="Directory for CSV/JSON result artifacts"
    )
    p.add_argument(
        "--report_path", type=str, default="results/summary.txt", help="Write numeric summary here"
    )
    p.add_argument(
        "--resume_existing",
        action="store_true",
        help="Reuse existing checkpoints in out_root when present.",
    )
    p.add_argument(
        "--device", type=str, default=None, help="Override COMMON_KNOBS.device (auto|cpu|cuda)"
    )
    p.add_argument("--seed", type=int, default=None, help="Override COMMON_KNOBS.seed")
    p.add_argument("--batch_size", type=int, default=None, help="Override COMMON_KNOBS.batch_size")
    p.add_argument(
        "--seq_len",
        type=int,
        default=None,
        help="Override COMMON_KNOBS.seq_len (truncate if needed)",
    )
    p.add_argument(
        "--test_ratio", type=float, default=None, help="Override COMMON_KNOBS.test_ratio"
    )
    p.add_argument(
        "--ref_estimator", type=str, default="iptw_msm", choices=["iptw_msm", "gformula"]
    )
    p.add_argument(
        "--do_horizon", type=int, default=-1, help="Do/policy horizon (-1 uses seq_len-1)."
    )
    p.add_argument(
        "--oracle_mc_samples",
        type=int,
        default=200,
        help="MC samples for OracleIRT (irt_synth only).",
    )
    p.add_argument(
        "--oracle_mc_seed", type=int, default=0, help="MC seed for OracleIRT (irt_synth only)."
    )
    p.add_argument(
        "--oracle_mc_batch",
        type=int,
        default=256,
        help="MC batch size for OracleIRT (irt_synth only).",
    )
    p.add_argument(
        "--debug_oracle", action="store_true", default=False, help="Run OracleIRT sanity checks."
    )
    p.add_argument("--t0_list", type=str, default="0,5,10")
    p.add_argument("--actions", type=str, default="0,1")
    p.add_argument("--subgroups", type=str, default="all,low,high")
    p.add_argument("--policy_set", type=str, default="fixed", choices=["fixed", "ablation"])
    p.add_argument("--eval_tstr", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--predictors", type=str, default="logreg,mlp,sakt")
    p.add_argument("--eval_calibration", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--eval_privacy", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--disc_input", type=str, default="logits", choices=["logits", "sampled"])
    p.add_argument(
        "--k_c_ratio", type=float, default=None, help="CausalSeqGAN: ratio of causal latent dims."
    )
    p.add_argument(
        "--w_do", type=float, default=None, help="CausalSeqGAN: do-alignment loss weight."
    )
    p.add_argument(
        "--w_cf",
        type=float,
        default=None,
        help="CausalSeqGAN: counterfactual consistency loss weight.",
    )
    p.add_argument(
        "--w_advT_causal", type=float, default=None, help="CausalSeqGAN: adversarial T loss on K_c."
    )
    p.add_argument(
        "--w_t_pred_spurious",
        type=float,
        default=None,
        help="CausalSeqGAN: T prediction loss on K_s.",
    )
    p.add_argument(
        "--do_horizon_train",
        type=int,
        default=None,
        help="CausalSeqGAN: horizon used inside do-alignment training loss.",
    )
    p.add_argument(
        "--causal_every",
        type=int,
        default=None,
        help="CausalSeqGAN: compute causal losses every N steps.",
    )
    p.add_argument(
        "--cf_every",
        type=int,
        default=None,
        help="CausalSeqGAN: compute counterfactual loss every N steps.",
    )
    p.add_argument(
        "--epochs_a", type=int, default=None, help="CausalSeqGAN: Stage A supervised epochs."
    )
    p.add_argument(
        "--epochs_b", type=int, default=None, help="CausalSeqGAN: Stage B adversarial epochs."
    )
    p.add_argument(
        "--epochs_c", type=int, default=None, help="CausalSeqGAN: Stage C causal epochs."
    )
    p.add_argument(
        "--plot_cf", action="store_true", default=True, help="Auto-generate counterfactual plots."
    )
    p.add_argument("--no_plot_cf", action="store_false", dest="plot_cf")
    args = p.parse_args(arguments)

    datasets = _parse_str_list(args.datasets, default=ACTIVE_DATASETS)
    if args.baseline is not None:
        models = [str(args.baseline).strip()]
    else:
        models = _parse_str_list(args.models, default=[CAUSALSEQGAN_ID])
    models = [
        CAUSALSEQGAN_ID if name.casefold() == CAUSALSEQGAN_NAME.casefold() else name
        for name in models
    ]
    unknown_models = set(models) - set(MODEL_KNOBS)
    if unknown_models:
        p.error("Unknown models: " + ", ".join(sorted(unknown_models)))
    dataset_paths = dict(DATASET_PATHS)
    if args.data_npz is not None:
        if args.datasets is not None:
            p.error("Use either --data-npz or --datasets")
        dataset_paths["custom"] = str(args.data_npz)
        datasets = ["custom"]

    common = dict(COMMON_KNOBS)
    if args.device is not None:
        common["device"] = str(args.device)
    if args.seed is not None:
        common["seed"] = int(args.seed)
    if args.batch_size is not None:
        common["batch_size"] = int(args.batch_size)
    if args.seq_len is not None:
        common["seq_len"] = int(args.seq_len)
    if args.test_ratio is not None:
        common["test_ratio"] = float(args.test_ratio)

    _set_seed(int(common["seed"]))
    device = _device_from_arg(str(common["device"]))

    out_root = _resolve_path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    results_dir = _resolve_path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    report_path = _resolve_path(args.report_path)

    t0_list = _parse_int_list(args.t0_list, default=[0, 5, 10])
    actions = _parse_int_list(args.actions, default=[0, 1])
    subgroup_names = _parse_str_list(args.subgroups, default=["all", "low", "high"])
    predictors = _parse_str_list(args.predictors, default=["logreg", "mlp", "sakt"])

    dr_frames = []
    dr_summary_frames = []
    factual_auc_frames = []
    oracle_ite_frames = []
    oracle_policy_frames = []
    tstr_frames = []
    leakage_frames = []
    nn_frames = []
    mia_frames = []
    privacy_reports = []
    run_config_rows = []
    max_horizon = 0

    manifest = {
        "run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "dataset": ",".join(datasets),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "artifacts": [],
    }

    for ds_name in datasets:
        if ds_name not in dataset_paths:
            raise KeyError(
                f"Unknown dataset key: {ds_name}. Available: {sorted(dataset_paths.keys())}"
            )
        ds_path = _resolve_path(dataset_paths[ds_name])
        dataset_knobs = DATASET_SPECIFIC_KNOBS.get(ds_name, {})
        common_ds = dict(common)
        for key in ("batch_size", "seq_len", "device", "seed", "test_ratio"):
            if key in dataset_knobs:
                common_ds[key] = dataset_knobs[key]
        if dataset_knobs:
            print(f"Dataset settings: {ds_name}")
        ds_raw = NPZSequenceDataset(ds_path)
        desired_seq_len = int(common_ds.get("seq_len", ds_raw.seq_len))
        if desired_seq_len < int(ds_raw.seq_len):
            ds_raw.X = ds_raw.X
            ds_raw.A = ds_raw.A[:, :desired_seq_len]
            ds_raw.T = ds_raw.T[:, :desired_seq_len]
            ds_raw.Y = ds_raw.Y[:, :desired_seq_len]
            ds_raw.M = ds_raw.M[:, :desired_seq_len]
            ds_raw.seq_len = desired_seq_len

        full_batch = _batch_from_dataset(ds_raw)
        train_idx, test_idx = _split_indices(
            len(ds_raw), test_ratio=float(common_ds["test_ratio"]), seed=int(common_ds["seed"])
        )
        train_batch = _subset_batch(full_batch, train_idx)
        test_batch = _subset_batch(full_batch, test_idx)
        subgroups = _build_subgroups(test_batch, subgroup_names)
        for sg in subgroups:
            sg["dataset"] = ds_name

        ds_knobs = DATASET_SPECIFIC_KNOBS.get(ds_name, {})
        estimator_type = str(ds_knobs.get("ref_estimator", args.ref_estimator))
        print(f"[{ds_name}] Using ref_estimator: {estimator_type}")
        if estimator_type == "iptw_msm":
            ref_estimator = IPTWMSM(clip=10.0)
        elif estimator_type == "gformula":
            ref_estimator = GFormula()
        else:
            raise ValueError(f"Unknown ref_estimator: {estimator_type}")
        ref_estimator.fit(train_batch)

        oracle_estimator = None
        if ds_name == "irt_synth":
            try:
                oracle_estimator = PrecomputedOracleEstimator(str(ds_path), device=str(device))
                print(f"[oracle] Loaded precomputed oracle from {ds_path}")
            except Exception as e:
                oracle_estimator = None
                print(f"[oracle] Precomputed oracle unavailable: {e}")

        if isinstance(ref_estimator, IPTWMSM):
            propensity_estimator = ref_estimator
        else:
            propensity_estimator = IPTWMSM(clip=10.0)
            propensity_estimator.fit(train_batch)
        has_oracle = bool(getattr(ref_estimator, "is_oracle", False))
        if has_oracle and bool(args.debug_oracle) and hasattr(ref_estimator, "debug_sanity_check"):
            ref_estimator.debug_sanity_check(test_batch)

        train_ds = TrajectoryDataset(
            X=train_batch.X,
            A=train_batch.A,
            T=train_batch.T,
            Y=train_batch.Y,
            mask=train_batch.mask,
        )
        train_dl = _make_dataloader(
            train_ds, batch_size=int(common_ds["batch_size"]), shuffle=True, drop_last=True
        )

        meta = _dataset_meta(ds_raw)
        do_horizon = _resolve_horizon(args.do_horizon, seq_len=int(meta["seq_len"]))
        max_horizon = max(max_horizon, do_horizon)

        ckpt_paths: dict[str, Path] = {}
        for model_key in models:
            knobs = dict(MODEL_KNOBS.get(model_key, {}))
            knobs["save_checkpoints"] = bool(common_ds.get("keep_checkpoints", True))
            knobs["disc_input"] = str(args.disc_input)
            knobs["do_horizon"] = int(do_horizon)
            knobs["ref_estimator"] = ref_estimator
            knobs["seed"] = int(common_ds["seed"])
            if dataset_knobs:
                model_dataset_knobs = dict(dataset_knobs)
                model_dataset_knobs.pop("ref_estimator", None)
                if model_key != CAUSALSEQGAN_ID:
                    for key in ("w_do", "w_advT_causal", "w_cf", "w_t_pred_spurious"):
                        model_dataset_knobs.pop(key, None)
                knobs.update(model_dataset_knobs)
                knobs["ref_estimator"] = ref_estimator
            if model_key == CAUSALSEQGAN_ID:
                if args.k_c_ratio is not None:
                    knobs["k_c_ratio"] = float(args.k_c_ratio)
                if args.w_do is not None:
                    knobs["w_do"] = float(args.w_do)
                if args.w_cf is not None:
                    knobs["w_cf"] = float(args.w_cf)
                if args.w_advT_causal is not None:
                    knobs["w_advT_causal"] = float(args.w_advT_causal)
                if args.w_t_pred_spurious is not None:
                    knobs["w_t_pred_spurious"] = float(args.w_t_pred_spurious)
                if args.do_horizon_train is not None:
                    knobs["do_horizon_train"] = int(args.do_horizon_train)
                if args.causal_every is not None:
                    knobs["causal_every"] = int(args.causal_every)
                if args.cf_every is not None:
                    knobs["cf_every"] = int(args.cf_every)
                if args.epochs_a is not None:
                    knobs["epochs_a"] = int(args.epochs_a)
                if args.epochs_b is not None:
                    knobs["epochs_b"] = int(args.epochs_b)
                if args.epochs_c is not None:
                    knobs["epochs_c"] = int(args.epochs_c)
            is_causalseqgan = model_key == CAUSALSEQGAN_ID
            do_horizon_train = (
                int(knobs.get("do_horizon_train", do_horizon)) if is_causalseqgan else np.nan
            )
            causal_every = int(knobs.get("causal_every", 20)) if is_causalseqgan else np.nan
            cf_every = (
                int(knobs.get("cf_every", knobs.get("cf_interval", 20)))
                if is_causalseqgan
                else np.nan
            )
            run_config_rows.append(
                {
                    "model": model_key,
                    "dataset": ds_name,
                    "do_horizon_train": do_horizon_train,
                    "do_horizon_eval": int(do_horizon),
                    "causal_every": causal_every,
                    "cf_every": cf_every,
                    "k_c_ratio": float(knobs.get("k_c_ratio", np.nan))
                    if is_causalseqgan
                    else np.nan,
                    "w_do": float(knobs.get("w_do", np.nan)) if is_causalseqgan else np.nan,
                    "w_advT_causal": float(knobs.get("w_advT_causal", np.nan))
                    if is_causalseqgan
                    else np.nan,
                    "w_t_pred_spurious": float(knobs.get("w_t_pred_spurious", np.nan))
                    if is_causalseqgan
                    else np.nan,
                    "w_cf": float(knobs.get("w_cf", np.nan)) if is_causalseqgan else np.nan,
                    "epochs_a": int(knobs.get("epochs_a", 0)) if is_causalseqgan else np.nan,
                    "epochs_b": int(knobs.get("epochs_b", 0)) if is_causalseqgan else np.nan,
                    "epochs_c": int(knobs.get("epochs_c", 0)) if is_causalseqgan else np.nan,
                }
            )

            run_dir = out_root / ds_name / model_key / f"seed{int(common_ds['seed'])}"
            run_dir.mkdir(parents=True, exist_ok=True)

            ckpt_path = (
                latest_checkpoint(run_dir, model_key) if bool(args.resume_existing) else None
            )
            if ckpt_path is not None:
                print(f"[resume] {ds_name}/{model_key}: using {ckpt_path.as_posix()}")
            elif model_key in {CAUSALSEQGAN_ID, "rcgan"}:
                ckpt_path = _train_adversarial_generator(
                    model_key,
                    train_dl=train_dl,
                    meta=meta,
                    device=device,
                    out_dir=run_dir,
                    knobs=knobs,
                )
            elif model_key == "crn":
                ckpt_path = _train_crn(
                    train_dl=train_dl, meta=meta, device=device, out_dir=run_dir, knobs=knobs
                )
            elif model_key == "vae":
                ckpt_path = _train_vae(
                    train_dl=train_dl, meta=meta, device=device, out_dir=run_dir, knobs=knobs
                )
            elif model_key == "diffusion":
                ckpt_path = _train_diffusion(
                    train_dl=train_dl, meta=meta, device=device, out_dir=run_dir, knobs=knobs
                )
            elif model_key == "timegan":
                ckpt_path = _train_timegan(
                    train_dl=train_dl, meta=meta, device=device, out_dir=run_dir, knobs=knobs
                )
            else:
                raise ValueError(f"Unknown model={model_key}")

            ckpt = torch.load(ckpt_path, map_location=device)
            gen_model = load_base_model_from_checkpoint(ckpt, device=device)
            gen_model.name = CAUSALSEQGAN_NAME if model_key == CAUSALSEQGAN_ID else model_key
            ckpt_paths[model_key] = ckpt_path

            try:
                leakage_auc = _compute_t_leakage_auc(
                    gen_model,
                    train_batch=train_batch,
                    test_batch=test_batch,
                    max_points=20000,
                    epochs=3,
                )
            except Exception as e:
                leakage_auc = float("nan")
                add_artifact(
                    manifest,
                    kind="table",
                    model=model_key,
                    dataset=ds_name,
                    path="results/leakage_auc.csv",
                    meta={"supported": False, "skip_reason": str(e)},
                )
            leakage_frames.append(
                pd.DataFrame(
                    [
                        {
                            "model": model_key,
                            "dataset": ds_name,
                            "t_leakage_auc": leakage_auc,
                        }
                    ]
                )
            )
            try:
                df_dr, dr_summary = compute_dr_policy_values(
                    gen_model=gen_model,
                    propensity_estimator=propensity_estimator,
                    data=test_batch,
                    actions=actions,
                    horizon=int(do_horizon),
                    dataset=ds_name,
                    seed=int(common_ds["seed"]),
                    policy_set=str(args.policy_set),
                )
                dr_frames.append(df_dr)
                dr_summary_frames.append(pd.DataFrame([dr_summary]))
                if dr_summary.get("policy_supported") == 0 and dr_summary.get("policy_skip_reason"):
                    add_artifact(
                        manifest,
                        kind="policy_curve",
                        model=model_key,
                        dataset=ds_name,
                        path="results/dr_policy_values.csv",
                        meta={
                            "supported": False,
                            "skip_reason": dr_summary.get("policy_skip_reason", ""),
                        },
                    )
            except Exception as e:
                add_artifact(
                    manifest,
                    kind="table",
                    model=model_key,
                    dataset=ds_name,
                    path="results/dr_policy_values.csv",
                    meta={"supported": False, "skip_reason": str(e)},
                )
                dr_frames.append(
                    pd.DataFrame(
                        [
                            {
                                "model": model_key,
                                "dataset": ds_name,
                                "policy": "unknown",
                                "horizon": int(do_horizon),
                                "dr_value": np.nan,
                                "dm_value": np.nan,
                                "correction": np.nan,
                                "w_mean": np.nan,
                                "w_max": np.nan,
                                "supported": 0.0,
                                "skip_reason": str(e),
                            }
                        ]
                    )
                )
                dr_summary_frames.append(
                    pd.DataFrame(
                        [
                            {
                                "model": model_key,
                                "dataset": ds_name,
                                "horizon": int(do_horizon),
                                "dr_value_mean": np.nan,
                                "dm_value_mean": np.nan,
                                "correction_mean": np.nan,
                                "policy_supported": 0.0,
                                "policy_skip_reason": str(e),
                            }
                        ]
                    )
                )
            try:
                factual_auc, factual_n, factual_skip = _compute_factual_auc(
                    gen_model, data=test_batch, batch_size=int(common_ds.get("batch_size", 128))
                )
                supported = 0.0 if factual_skip else 1.0
            except Exception as e:
                factual_auc, factual_n, factual_skip = float("nan"), 0, str(e)
                supported = 0.0
                add_artifact(
                    manifest,
                    kind="table",
                    model=model_key,
                    dataset=ds_name,
                    path="results/factual_auc.csv",
                    meta={"supported": False, "skip_reason": str(e)},
                )
            factual_auc_frames.append(
                pd.DataFrame(
                    [
                        {
                            "model": model_key,
                            "dataset": ds_name,
                            "factual_auc": factual_auc,
                            "n_samples": int(factual_n),
                            "supported": supported,
                            "skip_reason": factual_skip,
                        }
                    ]
                )
            )
            try:
                df_oracle_ite, df_oracle_policy = compute_oracle_metrics(
                    gen_model=gen_model,
                    oracle_estimator=oracle_estimator,
                    data=test_batch,
                    dataset_name=ds_name,
                    t0_list=t0_list,
                    horizon=int(do_horizon),
                    actions=actions,
                    subgroups=subgroups,
                    policy_set=str(args.policy_set),
                    seed=int(common_ds["seed"]),
                    n_gen=20,
                )
            except Exception as e:
                df_oracle_ite = pd.DataFrame(
                    [
                        {
                            "model": model_key,
                            "dataset": ds_name,
                            "t0": int(t0_list[0]) if t0_list else 0,
                            "subgroup": "all",
                            "horizon": int(do_horizon),
                            "supported": False,
                            "skip_reason": str(e),
                        }
                    ]
                )
                df_oracle_policy = pd.DataFrame(
                    [
                        {
                            "model": model_key,
                            "dataset": ds_name,
                            "policy": "unknown",
                            "horizon": int(do_horizon),
                            "supported": False,
                            "skip_reason": str(e),
                        }
                    ]
                )

            if df_oracle_ite is not None:
                oracle_ite_frames.append(df_oracle_ite)
            if df_oracle_policy is not None:
                oracle_policy_frames.append(df_oracle_policy)

            if args.eval_tstr:
                for predictor in predictors:
                    try:
                        df_tstr = run_tstr_trts(
                            gen_model=gen_model,
                            real_train=train_batch,
                            real_test=test_batch,
                            n_synth=min(2048, int(train_batch.X.shape[0])),
                            predictor=predictor,
                            seed=int(common_ds["seed"]),
                            save_calibration=bool(args.eval_calibration),
                        )
                        df_tstr["dataset"] = ds_name
                        tstr_frames.append(df_tstr)
                    except Exception as e:
                        add_artifact(
                            manifest,
                            kind="table",
                            model=model_key,
                            dataset=ds_name,
                            path="results/tstr_trts.csv",
                            meta={"supported": False, "skip_reason": str(e)},
                        )
                        df_fallback = pd.DataFrame(
                            [
                                {
                                    "model": model_key,
                                    "dataset": ds_name,
                                    "predictor": predictor,
                                    "setting": "TRTS",
                                    "auc": np.nan,
                                    "rmse": np.nan,
                                    "brier": np.nan,
                                    "ece": np.nan,
                                    "n_train": int(train_batch.X.shape[0]),
                                    "n_test": int(test_batch.X.shape[0]),
                                }
                            ]
                        )
                        tstr_frames.append(df_fallback)

            if args.eval_privacy:
                try:
                    publish_mode = "y_only"
                    embed_space = "y_only"
                    perturb = {"x_noise_scale": 0.0, "a_flip_prob": 0.0, "t_flip_prob": 0.0}
                    rng = np.random.default_rng(int(common_ds["seed"]))
                    n_synth_total = min(1024, int(train_batch.X.shape[0]))
                    train_idx = rng.integers(0, train_batch.X.shape[0], size=n_synth_total)
                    synth_source = _subset_batch(train_batch, train_idx.tolist())

                    ro = gen_model.rollout(synth_source, do_t=None, policy=None, horizon=None, t0=0)
                    y_prob = ro["Y_prob"].detach().cpu().numpy()
                    y_sample = rng.binomial(n=1, p=np.clip(y_prob, 1e-4, 1.0 - 1e-4)).astype(
                        np.float32
                    )
                    synth_batch = TrajectoryBatch(
                        X=torch.zeros_like(synth_source.X),
                        A=torch.zeros_like(synth_source.A),
                        T=torch.zeros_like(synth_source.T),
                        Y=torch.as_tensor(y_sample),
                        mask=synth_source.mask,
                        lengths=synth_source.lengths,
                    )
                    nn_df = compute_nn_distance(
                        real=train_batch,
                        synth=synth_batch,
                        metric="embedding_l2",
                        embed_space=embed_space,
                    )
                    nn_df["model"] = model_key
                    nn_df["dataset"] = ds_name
                    nn_frames.append(nn_df)

                    attack_features = ["recon_error", "avg_confidence"]
                    mia = run_membership_inference(
                        gen_model=gen_model,
                        real_train=train_batch,
                        real_holdout=test_batch,
                        attack_features=attack_features,
                        embed_space=embed_space,
                        seed=int(common_ds["seed"]),
                    )
                    mia_df = mia_to_dataframe(
                        mia,
                        model=model_key,
                        dataset=ds_name,
                        attack_features=attack_features,
                        n_in=int(train_batch.X.shape[0]),
                        n_out=int(test_batch.X.shape[0]),
                    )
                    mia_frames.append(mia_df)

                    distances = nn_df["distance"].to_numpy(dtype=np.float64)
                    report = {
                        "dataset": ds_name,
                        "model": model_key,
                        "nn_distance": {
                            "metric": "embedding_l2",
                            "mean": float(np.mean(distances)) if distances.size else float("nan"),
                            "p01": float(np.quantile(distances, 0.01))
                            if distances.size
                            else float("nan"),
                            "p05": float(np.quantile(distances, 0.05))
                            if distances.size
                            else float("nan"),
                            "p10": float(np.quantile(distances, 0.10))
                            if distances.size
                            else float("nan"),
                        },
                        "publish_mode": publish_mode,
                        "perturbation": perturb,
                        "mia": {
                            "attack_auc": float(mia.get("attack_auc", np.nan)),
                            "attack_acc": float(mia.get("attack_acc", np.nan)),
                            "attack_balanced_acc": float(mia.get("attack_balanced_acc", np.nan)),
                            "features": attack_features,
                        },
                    }
                    privacy_reports.append(report)
                except Exception as e:
                    add_artifact(
                        manifest,
                        kind="table",
                        model=model_key,
                        dataset=ds_name,
                        path="results/privacy_report.json",
                        meta={"supported": False, "skip_reason": str(e)},
                    )
                    nn_frames.append(
                        pd.DataFrame(
                            [
                                {
                                    "model": model_key,
                                    "dataset": ds_name,
                                    "metric": "embedding_l2",
                                    "synth_id": -1,
                                    "nn_real_id": -1,
                                    "distance": np.nan,
                                }
                            ]
                        )
                    )
                    mia_frames.append(
                        pd.DataFrame(
                            [
                                {
                                    "model": model_key,
                                    "dataset": ds_name,
                                    "attack_features": "y_stats,nn_distance",
                                    "attack_auc": np.nan,
                                    "attack_acc": np.nan,
                                    "attack_balanced_acc": np.nan,
                                    "n_in": int(train_batch.X.shape[0]),
                                    "n_out": int(test_batch.X.shape[0]),
                                }
                            ]
                        )
                    )
                    privacy_reports.append(
                        {
                            "dataset": ds_name,
                            "model": model_key,
                            "nn_distance": {
                                "metric": "embedding_l2",
                                "mean": float("nan"),
                                "p01": float("nan"),
                                "p05": float("nan"),
                                "p10": float("nan"),
                            },
                            "publish_mode": "y_only",
                            "perturbation": {
                                "x_noise_scale": 0.0,
                                "a_flip_prob": 0.0,
                                "t_flip_prob": 0.0,
                            },
                            "mia": {
                                "attack_auc": float("nan"),
                                "attack_acc": float("nan"),
                                "attack_balanced_acc": float("nan"),
                                "features": ["y_stats", "nn_distance"],
                            },
                        }
                    )

        cf_path = None
        if args.plot_cf:
            cf_path = _maybe_plot_cf_consistency_multi(
                ckpt_paths=ckpt_paths,
                model_order=[str(m) for m in models],
                test_batch=test_batch,
                dataset=ds_name,
                t0_list=t0_list,
                out_dir=results_dir / "figures",
                seed=int(common_ds["seed"]),
                actions=actions,
                plot_horizon=min(20, int(meta.get("seq_len", 1)) - 1),
                device=device,
            )
        if cf_path is not None:
            add_artifact(
                manifest,
                kind="plot",
                model="all",
                dataset=ds_name,
                path=str(Path(cf_path).as_posix()),
                meta={"title": "counterfactual_bifurcation_multi"},
            )
            cf_data = Path(cf_path).with_suffix(".npz")
            if cf_data.exists():
                add_artifact(
                    manifest,
                    kind="data",
                    model="all",
                    dataset=ds_name,
                    path=str(cf_data.as_posix()),
                    meta={"title": "counterfactual_bifurcation_multi_data"},
                )

    dr_df = pd.concat(dr_frames, ignore_index=True) if dr_frames else pd.DataFrame()
    dr_summary_df = (
        pd.concat(dr_summary_frames, ignore_index=True) if dr_summary_frames else pd.DataFrame()
    )
    factual_auc_df = (
        pd.concat(factual_auc_frames, ignore_index=True) if factual_auc_frames else pd.DataFrame()
    )
    oracle_ite_df = (
        pd.concat(oracle_ite_frames, ignore_index=True) if oracle_ite_frames else pd.DataFrame()
    )
    oracle_policy_df = (
        pd.concat(oracle_policy_frames, ignore_index=True)
        if oracle_policy_frames
        else pd.DataFrame()
    )
    tstr_df = pd.concat(tstr_frames, ignore_index=True) if tstr_frames else pd.DataFrame()
    leakage_df = pd.concat(leakage_frames, ignore_index=True) if leakage_frames else pd.DataFrame()
    nn_df = pd.concat(nn_frames, ignore_index=True) if nn_frames else pd.DataFrame()
    mia_df = pd.concat(mia_frames, ignore_index=True) if mia_frames else pd.DataFrame()

    dr_cols = [
        "model",
        "dataset",
        "policy",
        "horizon",
        "dr_value",
        "dm_value",
        "correction",
        "w_mean",
        "w_max",
        "supported",
        "skip_reason",
    ]
    if dr_df.empty:
        dr_df = pd.DataFrame(columns=dr_cols)
    else:
        dr_df = dr_df.reindex(columns=dr_cols + [c for c in dr_df.columns if c not in dr_cols])

    dr_summary_cols = [
        "model",
        "dataset",
        "horizon",
        "dr_value_mean",
        "dm_value_mean",
        "correction_mean",
        "policy_supported",
        "policy_skip_reason",
    ]
    if dr_summary_df.empty:
        dr_summary_df = pd.DataFrame(columns=dr_summary_cols)
    else:
        dr_summary_df = dr_summary_df.reindex(
            columns=dr_summary_cols + [c for c in dr_summary_df.columns if c not in dr_summary_cols]
        )

    factual_auc_cols = ["model", "dataset", "factual_auc", "n_samples", "supported", "skip_reason"]
    if factual_auc_df.empty:
        factual_auc_df = pd.DataFrame(columns=factual_auc_cols)
    else:
        factual_auc_df = factual_auc_df.reindex(
            columns=factual_auc_cols
            + [c for c in factual_auc_df.columns if c not in factual_auc_cols]
        )

    oracle_ite_cols = [
        "model",
        "dataset",
        "t0",
        "subgroup",
        "horizon",
        "pehe",
        "ite_rmse",
        "ate_true",
        "ate_hat",
        "ate_abs_error",
        "ate_rmse",
        "n_effective",
        "tau_true_mean",
        "tau_hat_mean",
        "n_gen",
        "supported",
        "skip_reason",
    ]
    if oracle_ite_df.empty:
        oracle_ite_df = pd.DataFrame(columns=oracle_ite_cols)
    else:
        oracle_ite_df = oracle_ite_df.reindex(
            columns=oracle_ite_cols + [c for c in oracle_ite_df.columns if c not in oracle_ite_cols]
        )

    oracle_policy_cols = [
        "model",
        "dataset",
        "policy",
        "horizon",
        "oracle_value",
        "gen_value",
        "value_abs_error",
        "regret_oracle",
        "supported",
        "skip_reason",
    ]
    if oracle_policy_df.empty:
        oracle_policy_df = pd.DataFrame(columns=oracle_policy_cols)
    else:
        oracle_policy_df = oracle_policy_df.reindex(
            columns=oracle_policy_cols
            + [c for c in oracle_policy_df.columns if c not in oracle_policy_cols]
        )

    tstr_cols = [
        "model",
        "dataset",
        "predictor",
        "setting",
        "auc",
        "rmse",
        "brier",
        "ece",
        "n_train",
        "n_test",
    ]
    if tstr_df.empty:
        tstr_df = pd.DataFrame(columns=tstr_cols)
    else:
        tstr_df = tstr_df.reindex(columns=tstr_cols)

    leakage_cols = ["model", "dataset", "t_leakage_auc"]
    if leakage_df.empty:
        leakage_df = pd.DataFrame(columns=leakage_cols)
    else:
        leakage_df = leakage_df.reindex(columns=leakage_cols)

    nn_cols = ["model", "dataset", "metric", "synth_id", "nn_real_id", "distance"]
    if nn_df.empty:
        nn_df = pd.DataFrame(columns=nn_cols)
    else:
        nn_df = nn_df.reindex(columns=nn_cols)

    mia_cols = [
        "model",
        "dataset",
        "attack_features",
        "attack_auc",
        "attack_acc",
        "attack_balanced_acc",
        "n_in",
        "n_out",
    ]
    if mia_df.empty:
        mia_df = pd.DataFrame(columns=mia_cols)
    else:
        mia_df = mia_df.reindex(columns=mia_cols)

    dr_df.to_csv(results_dir / "dr_policy_values.csv", index=False)
    dr_summary_df.to_csv(results_dir / "dr_policy_summary.csv", index=False)
    factual_auc_df.to_csv(results_dir / "factual_auc.csv", index=False)
    oracle_ite_df.to_csv(results_dir / "oracle_ite_metrics.csv", index=False)
    oracle_policy_df.to_csv(results_dir / "oracle_policy_metrics.csv", index=False)
    tstr_df.to_csv(results_dir / "tstr_trts.csv", index=False)
    leakage_df.to_csv(results_dir / "leakage_auc.csv", index=False)
    nn_df.to_csv(results_dir / "privacy_nn_distance.csv", index=False)
    mia_df.to_csv(results_dir / "privacy_mia.csv", index=False)

    with (results_dir / "privacy_report.json").open("w", encoding="utf-8") as f:
        json.dump(privacy_reports, f, indent=2)

    add_artifact(
        manifest,
        kind="table",
        model="all",
        dataset=",".join(datasets),
        path="results/dr_policy_values.csv",
        meta={},
    )
    add_artifact(
        manifest,
        kind="table",
        model="all",
        dataset=",".join(datasets),
        path="results/dr_policy_summary.csv",
        meta={},
    )
    add_artifact(
        manifest,
        kind="table",
        model="all",
        dataset=",".join(datasets),
        path="results/factual_auc.csv",
        meta={},
    )
    add_artifact(
        manifest,
        kind="table",
        model="all",
        dataset=",".join(datasets),
        path="results/oracle_ite_metrics.csv",
        meta={},
    )
    add_artifact(
        manifest,
        kind="table",
        model="all",
        dataset=",".join(datasets),
        path="results/oracle_policy_metrics.csv",
        meta={},
    )

    for ds_name in datasets:
        suffix = f"_{ds_name}"
        if not dr_df.empty and "dataset" in dr_df.columns:
            dr_df[dr_df["dataset"] == ds_name].to_csv(
                results_dir / f"dr_policy_values{suffix}.csv", index=False
            )
        if not dr_summary_df.empty and "dataset" in dr_summary_df.columns:
            dr_summary_df[dr_summary_df["dataset"] == ds_name].to_csv(
                results_dir / f"dr_policy_summary{suffix}.csv", index=False
            )
        if not factual_auc_df.empty and "dataset" in factual_auc_df.columns:
            factual_auc_df[factual_auc_df["dataset"] == ds_name].to_csv(
                results_dir / f"factual_auc{suffix}.csv", index=False
            )
        if not oracle_ite_df.empty and "dataset" in oracle_ite_df.columns:
            oracle_ite_df[oracle_ite_df["dataset"] == ds_name].to_csv(
                results_dir / f"oracle_ite_metrics{suffix}.csv", index=False
            )
        if not oracle_policy_df.empty and "dataset" in oracle_policy_df.columns:
            oracle_policy_df[oracle_policy_df["dataset"] == ds_name].to_csv(
                results_dir / f"oracle_policy_metrics{suffix}.csv", index=False
            )
        if not tstr_df.empty and "dataset" in tstr_df.columns:
            tstr_df[tstr_df["dataset"] == ds_name].to_csv(
                results_dir / f"tstr_trts{suffix}.csv", index=False
            )
        if not leakage_df.empty and "dataset" in leakage_df.columns:
            leakage_df[leakage_df["dataset"] == ds_name].to_csv(
                results_dir / f"leakage_auc{suffix}.csv", index=False
            )
        if not nn_df.empty and "dataset" in nn_df.columns:
            nn_df[nn_df["dataset"] == ds_name].to_csv(
                results_dir / f"privacy_nn_distance{suffix}.csv", index=False
            )
        if not mia_df.empty and "dataset" in mia_df.columns:
            mia_df[mia_df["dataset"] == ds_name].to_csv(
                results_dir / f"privacy_mia{suffix}.csv", index=False
            )
        report_ds = [r for r in privacy_reports if r.get("dataset") == ds_name]
        with (results_dir / f"privacy_report{suffix}.json").open("w", encoding="utf-8") as f:
            json.dump(report_ds, f, indent=2)

    privacy_summary_rows = []
    for report in privacy_reports:
        privacy_summary_rows.append(
            {
                "dataset": report["dataset"],
                "model": report["model"],
                "nn_mean": report["nn_distance"]["mean"],
                "nn_p01": report["nn_distance"]["p01"],
                "nn_p05": report["nn_distance"]["p05"],
                "nn_p10": report["nn_distance"]["p10"],
                "publish_mode": report.get("publish_mode", ""),
                "x_noise_scale": float(report.get("perturbation", {}).get("x_noise_scale", 0.0)),
                "a_flip_prob": float(report.get("perturbation", {}).get("a_flip_prob", 0.0)),
                "t_flip_prob": float(report.get("perturbation", {}).get("t_flip_prob", 0.0)),
                "attack_auc": report["mia"]["attack_auc"],
                "attack_acc": report["mia"]["attack_acc"],
                "attack_balanced_acc": report["mia"]["attack_balanced_acc"],
            }
        )
    privacy_summary_df = pd.DataFrame(privacy_summary_rows)
    run_config_df = pd.DataFrame(run_config_rows)
    run_config_manifest = run_config_df.where(pd.notnull(run_config_df), None).to_dict(
        orient="records"
    )
    manifest["run_config"] = run_config_manifest

    metrics_tables = {
        "dr_policy_values": dr_df,
        "dr_policy_summary": dr_summary_df,
        "factual_auc": factual_auc_df,
        "oracle_ite_metrics": oracle_ite_df,
        "oracle_policy_metrics": oracle_policy_df,
        "tstr_trts": tstr_df,
        "leakage_metrics": leakage_df,
        "privacy_summary": privacy_summary_df,
        "run_config": run_config_df,
    }
    write_results_summary(metrics_tables, str(report_path))
    report_base = Path(report_path)
    for ds_name in datasets:
        per_tables = {}
        for key, df in metrics_tables.items():
            if df is None or df.empty:
                per_tables[key] = df
            elif "dataset" in df.columns:
                per_tables[key] = df[df["dataset"] == ds_name].copy()
            else:
                per_tables[key] = df.copy()
        ds_report = report_base.with_name(f"{report_base.stem}_{ds_name}{report_base.suffix}")
        write_results_summary(per_tables, str(ds_report))

    run_visualization(
        results_dir=str(results_dir),
        out_dir=str(results_dir / "figures"),
        dataset=",".join(datasets),
        manifest_path=str(results_dir / "artifacts.json"),
        manifest=manifest,
        plot_calibration=bool(args.eval_calibration),
    )
    write_manifest(manifest, str(results_dir / "manifest.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
