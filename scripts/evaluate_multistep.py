"""Evaluate saved longitudinal predictors at multiple horizons without training."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines import load_base_model_from_checkpoint
from src.data import TrajectoryBatch
from src.training import _split_indices

MODELS = ["scm_causal", "crn", "timegan", "rcgan", "vae", "diffusion"]
LABELS = dict(zip(MODELS, ["CausalSeqGAN", "CRN", "TimeGAN", "RCGAN", "VAE", "Diffusion"]))
PATTERNS = dict(
    zip(
        MODELS,
        [
            "ckpt_scm_causal_C_ep20.pt",
            "ckpt_crn_ep30.pt",
            "ckpt_timegan.pt",
            "ckpt_rcgan_C_ep20.pt",
            "ckpt_vae_ep30.pt",
            "ckpt_diffusion_ep30.pt",
        ],
    )
)
SEQUENCE_LENGTH = 30
METRICS = ["pehe", "ate_abs_error", "ate_curve_rmse", "policy_value_error", "policy_regret"]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def checkpoint(model: str, seed: int, directory: Path) -> Path:
    matches = list((directory / model / f"seed{seed}").rglob(PATTERNS[model]))
    if len(matches) != 1:
        raise ValueError(f"Expected one final checkpoint: {model}, seed {seed}; found {matches}")
    return matches[0]


def mean_masked(values: np.ndarray, mask: np.ndarray, axis: int) -> np.ndarray:
    count = mask.sum(axis=axis)
    return (values * mask).sum(axis=axis) / np.maximum(count, 1.0)


def batch_from_arrays(arrays, ids: np.ndarray, device: torch.device) -> TrajectoryBatch:
    mask = torch.as_tensor(arrays["M"][ids, :SEQUENCE_LENGTH], dtype=torch.float32, device=device)
    return TrajectoryBatch(
        X=torch.as_tensor(arrays["X"][ids], dtype=torch.float32, device=device),
        A=torch.as_tensor(arrays["A"][ids, :SEQUENCE_LENGTH], device=device),
        T=torch.as_tensor(arrays["T"][ids, :SEQUENCE_LENGTH], dtype=torch.long, device=device),
        Y=torch.as_tensor(arrays["Y"][ids, :SEQUENCE_LENGTH], dtype=torch.float32, device=device),
        mask=mask,
        lengths=mask.sum(1).clamp(min=1).long(),
        ids=torch.as_tensor(ids, dtype=torch.long, device=device),
    )


def fixed_actions() -> tuple[list[str], np.ndarray]:
    t = np.arange(SEQUENCE_LENGTH)
    return ["never", "always", "early_on", "late_on"], np.array(
        [np.zeros_like(t), np.ones_like(t), (t < 10).astype(int), (t >= 10).astype(int)],
        dtype=np.int64,
    )


@torch.inference_mode()
def predict(model, batch, *, seed, t0s, n_gen):
    count = len(batch.X)
    do = np.empty((len(t0s), 2, count, SEQUENCE_LENGTH), np.float32)
    for index, t0 in enumerate(t0s):
        for action in [0, 1]:
            acc = np.zeros((count, SEQUENCE_LENGTH), np.float32)
            for mc in range(n_gen):
                torch.manual_seed(seed + 10007 * mc + 13 * t0 + 3 * action)
                output = model.rollout(
                    batch,
                    do_t={t0: action},
                    horizon=SEQUENCE_LENGTH - t0 - 1,
                    t0=t0,
                    teacher_forcing=False,
                )
                acc += output["Y_prob"].detach().cpu().numpy().astype(np.float32)
            do[index, action] = acc / float(n_gen)
    _, actions = fixed_actions()
    policy = np.empty((4, count, SEQUENCE_LENGTH), np.float32)
    for index, path in enumerate(actions):
        acc = np.zeros((count, SEQUENCE_LENGTH), np.float32)
        intervention = {t: int(value) for t, value in enumerate(path)}
        for mc in range(n_gen):
            torch.manual_seed(seed + 1000003 + 10007 * mc)
            output = model.rollout(
                batch, do_t=intervention, horizon=SEQUENCE_LENGTH - 1, t0=0, teacher_forcing=False
            )
            acc += output["Y_prob"].detach().cpu().numpy().astype(np.float32)
        policy[index] = acc / float(n_gen)
    return do, policy


def metrics_from_prediction(data, *, model: str, seed: int, steps: list[int]):
    mask = data["mask"].astype(np.float64)
    policy_rows, summary_rows, effect_rows = [], [], []
    for length in steps:
        m = mask[:, :length]
        true_values = np.array(
            [mean_masked(p[:, :length], m, axis=1).mean() for p in data["true_policy"]]
        )
        pred_values = np.array(
            [mean_masked(p[:, :length], m, axis=1).mean() for p in data["pred_policy"]]
        )
        chosen = int(np.argmax(pred_values))
        best = float(true_values.max())
        regret = best - float(true_values[chosen])
        optimal = [
            str(data["policy_names"][i])
            for i in np.flatnonzero(np.isclose(true_values, best, atol=1e-8, rtol=0))
        ]
        unique = len({tuple(path[:length]) for path in data["policy_actions"]})
        for index, name in enumerate(data["policy_names"]):
            policy_rows.append(
                dict(
                    model=model,
                    label=LABELS[model],
                    seed=seed,
                    steps=length,
                    policy=str(name),
                    oracle_value=true_values[index],
                    predicted_value=pred_values[index],
                    value_abs_error=abs(pred_values[index] - true_values[index]),
                    selected=index == chosen,
                    oracle_optimal=str(name) in optimal,
                    n_unique_action_prefixes=unique,
                )
            )
        summary_rows.append(
            dict(
                model=model,
                label=LABELS[model],
                seed=seed,
                steps=length,
                policy_value_error=float(np.mean(abs(pred_values - true_values))),
                policy_regret=regret,
                selected_policy=str(data["policy_names"][chosen]),
                oracle_optimal_set="|".join(optimal),
                n_unique_action_prefixes=unique,
                n_test=len(mask),
            )
        )
    for index, t0 in enumerate(data["do_t0s"].astype(int)):
        for length in steps:
            if length > SEQUENCE_LENGTH - t0:
                continue
            m = mask[:, t0 : t0 + length]
            p = data["pred_do"][index, :, :, t0 : t0 + length].astype(np.float64)
            truth = data["true_do"][index, :, :, :length].astype(np.float64)
            estimated, actual = p[1] - p[0], truth[1] - truth[0]
            tau_hat, tau_true = mean_masked(estimated, m, 1), mean_masked(actual, m, 1)
            curve_hat, curve_true = mean_masked(estimated, m, 0), mean_masked(actual, m, 0)
            effect_rows.append(
                dict(
                    model=model,
                    label=LABELS[model],
                    seed=seed,
                    t0=t0,
                    steps=length,
                    horizon=length - 1,
                    pehe=float(np.sqrt(np.mean((tau_hat - tau_true) ** 2))),
                    ate_abs_error=abs(float(tau_hat.mean() - tau_true.mean())),
                    ate_curve_rmse=float(np.sqrt(np.mean((curve_hat - curve_true) ** 2))),
                    ate_hat=float(tau_hat.mean()),
                    ate_true=float(tau_true.mean()),
                    n_test=len(mask),
                )
            )
    return pd.DataFrame(effect_rows), pd.DataFrame(summary_rows), pd.DataFrame(policy_rows)


def publish(output: Path):
    effects, policies, values, completed = [], [], [], []
    for path in sorted((output / "raw").glob("*/seed*/predictions.npz")):
        model, seed = path.parent.parent.name, int(path.parent.name[4:])
        with np.load(path, allow_pickle=False) as data:
            config = json.loads((path.parent / "manifest.json").read_text(encoding="utf-8"))
            effect, policy, value = metrics_from_prediction(
                data, model=model, seed=seed, steps=config["steps"]
            )
        completed.append(config)
        effects.append(effect)
        policies.append(policy)
        values.append(value)
    if not effects:
        return
    e, p, v = (
        pd.concat(effects, ignore_index=True),
        pd.concat(policies, ignore_index=True),
        pd.concat(values, ignore_index=True),
    )
    e.to_csv(output / "effects_by_seed.csv", index=False)
    p.to_csv(output / "policies_by_seed.csv", index=False)
    v.to_csv(output / "policy_values_by_seed.csv", index=False)
    combined = e.merge(p, on=["model", "label", "seed", "steps", "n_test"], validate="many_to_one")
    combined["policy_t0"] = 0
    combined.loc[combined.t0.ne(0), ["policy_value_error", "policy_regret"]] = np.nan
    combined.to_csv(output / "metrics_by_seed.csv", index=False)
    grouped = combined.groupby(["model", "label", "t0", "steps"])
    summary = grouped[METRICS].agg(["mean", "std"])
    summary.columns = [
        f"{name}_{'sd' if stat == 'std' else stat}" for name, stat in summary.columns
    ]
    summary = summary.reset_index().merge(grouped.seed.nunique().rename("n_seeds").reset_index())
    summary.to_csv(output / "summary.csv", index=False)
    index = dict(
        runs=len(completed),
        models=sorted(e.model.unique().tolist()),
        seeds=sorted(map(int, e.seed.unique())),
        complete_six_models_three_seeds=all(
            ((e.model == model) & (e.seed == seed)).any()
            for model in MODELS
            for seed in [42, 43, 44]
        ),
        effect_rows=len(e),
        policy_rows=len(p),
        per_policy_rows=len(v),
        run_manifests=completed,
    )
    (output / "completion_manifest.json").write_text(
        json.dumps(index, indent=2) + "\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["CausalSeqGAN", *MODELS[1:]],
        default=["CausalSeqGAN", *MODELS[1:]],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--steps", nargs="+", type=int, default=[1, 5, 10, 20, 30])
    parser.add_argument("--t0s", nargs="+", type=int, default=[0, 5, 10])
    parser.add_argument("--n-gen", type=int, default=20)
    parser.add_argument("--max-test", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("results/dkt"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--data", type=Path, default=Path("data/dkt_synth.npz"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/dkt"))
    parser.add_argument("--expected-data-sha256", help="Reject data that do not match this digest")
    args = parser.parse_args(argv)
    args.models = ["scm_causal" if name == "CausalSeqGAN" else name for name in args.models]
    if args.n_gen < 1 or any(x < 1 or x > SEQUENCE_LENGTH for x in args.steps):
        raise ValueError("Use positive Monte Carlo count and steps between 1 and 30")
    if any(x < 0 or x >= SEQUENCE_LENGTH for x in args.t0s):
        raise ValueError("Intervention times must be between 0 and 29")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    data_path = args.data.resolve()
    data_hash = digest(data_path)
    if args.expected_data_sha256 and data_hash != args.expected_data_sha256:
        raise ValueError("Unexpected oracle data hash")
    device = torch.device(args.device)
    torch.set_num_threads(2)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.30)
    arrays = np.load(data_path, allow_pickle=False)
    arrays = {
        name: arrays[name] for name in ["X", "A", "T", "Y", "M", "Y_cf", "Y_policy", "policy_names"]
    }
    policy_names, actions = fixed_actions()
    oracle_names = [x.decode() if isinstance(x, bytes) else str(x) for x in arrays["policy_names"]]
    global_manifest = dict(
        data_path=str(data_path),
        data_sha256=data_hash,
        sequence_length=SEQUENCE_LENGTH,
        steps=args.steps,
        t0s=args.t0s,
        n_gen=args.n_gen,
        policy_protocol="fixed 30-step paths; early-on switch10; late-on switch10; prefixes evaluated without rescaling switch",
        policy_prediction="uniform open-loop intervention interface for all six predictors, including Diffusion",
        policy_sampling="20 Monte Carlo draws by default, with common seed+1000003+10007*r across regimes",
        effect_sampling="original seed+10007*r+13*t0+3*action, with 20 draws by default",
        head="No RCT calibration head is applied; saved longitudinal generator checkpoints are evaluated directly",
        effect_estimand="single-time treatment substitution at t0 followed by factual treatment path; per-person temporal average",
        metric_definitions=dict(
            pehe="RMSE of each student's time-averaged effect",
            ate_abs_error="absolute population time-averaged effect error",
            ate_curve_rmse="RMSE of population effect curve across evaluated steps",
        ),
        invalid_steps="omitted when steps > 30-t0; no silent horizon truncation",
        training_performed=False,
        device=str(device),
    )
    (output / "evaluation_manifest.json").write_text(
        json.dumps(global_manifest, indent=2) + "\n", encoding="utf-8"
    )
    for model in args.models:
        for seed in args.seeds:
            run = output / "raw" / model / f"seed{seed}"
            run.mkdir(parents=True, exist_ok=True)
            path = checkpoint(model, seed, args.checkpoint_dir)
            ckpt_hash = digest(path)
            if args.resume and (run / "predictions.npz").exists():
                old = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
                if (
                    old["checkpoint_sha256"] == ckpt_hash
                    and old["n_gen"] == args.n_gen
                    and old["t0s"] == args.t0s
                    and old["max_test"] == args.max_test
                    and old["steps"] == args.steps
                ):
                    print(f"reuse {model}, seed={seed}", flush=True)
                    continue
            started = time.perf_counter()
            _, test_ids = _split_indices(len(arrays["X"]), test_ratio=0.2, seed=seed)
            ids = np.array(test_ids[: args.max_test] if args.max_test else test_ids, dtype=np.int64)
            batch = batch_from_arrays(arrays, ids, device)
            wrapper = load_base_model_from_checkpoint(
                torch.load(path, map_location="cpu", weights_only=False), device=device
            )
            print(f"{model}, seed={seed}, n={len(ids)}", flush=True)
            pred_do, pred_policy = predict(
                wrapper, batch, seed=seed, t0s=args.t0s, n_gen=args.n_gen
            )
            if not np.isfinite(pred_do).all() or not np.isfinite(pred_policy).all():
                raise ValueError("Non-finite generator predictions")
            true_do = np.full(pred_do.shape, np.nan, np.float32)
            for i, t0 in enumerate(args.t0s):
                length = SEQUENCE_LENGTH - t0
                true_do[i, :, :, :length] = np.moveaxis(arrays["Y_cf"][ids, t0, :, :length], 1, 0)
            true_policy = np.stack(
                [
                    arrays["Y_policy"][ids, oracle_names.index(name), :SEQUENCE_LENGTH]
                    for name in policy_names
                ]
            )
            np.savez_compressed(
                run / "predictions.npz",
                ids=ids,
                mask=batch.mask.cpu().numpy(),
                do_t0s=np.array(args.t0s),
                pred_do=pred_do,
                true_do=true_do,
                pred_policy=pred_policy,
                true_policy=true_policy,
                policy_names=np.array(policy_names),
                policy_actions=actions,
            )
            assert digest(path) == ckpt_hash
            record = dict(
                model=model,
                seed=seed,
                checkpoint=str(path),
                checkpoint_sha256=ckpt_hash,
                data_sha256=data_hash,
                n_gen=args.n_gen,
                t0s=args.t0s,
                steps=args.steps,
                max_test=args.max_test,
                n_test=len(ids),
                participant_ids_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),
                elapsed_seconds=time.perf_counter() - started,
                training_performed=False,
                predictions_sha256=digest(run / "predictions.npz"),
            )
            (run / "manifest.json").write_text(
                json.dumps(record, indent=2) + "\n", encoding="utf-8"
            )
            del wrapper, batch
            if device.type == "cuda":
                torch.cuda.empty_cache()
            publish(output)
            print(
                f"completed {model} seed={seed}, elapsed={record['elapsed_seconds']:.1f}s",
                flush=True,
            )
    publish(output)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        raise SystemExit(main())
