"""Shared training routines for trajectory models."""

from __future__ import annotations

import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.config import CAUSALSEQGAN_ID
from src.data import SequenceBatch, TrajectoryBatch, compute_lengths, move_batch
from src.model.baselines import (
    CRN,
    CRNConfig,
    RCGANConfig,
    RCGANGenerator,
    SeqDiffusion,
    SeqDiffusionConfig,
    SeqVAE,
    SeqVAEConfig,
    TimeGAN,
    TimeGANConfig,
)
from src.model.causalseqgan import CausalSeqGAN, CausalSeqGANConfig
from src.model.discriminators import SequenceDiscriminator, SequenceDiscriminatorConfig
from src.model.losses import (
    bernoulli_nll_from_logits,
    grad_reverse,
    mmd_rbf,
    treatment_ce_loss,
    wgan_g_loss,
    wgan_gp_d_loss,
)
from src.model.policy import TreatmentClassifier


def _device_from_arg(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _set_seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))


def _make_dataloader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    sampler: Any = None,
) -> DataLoader:
    def collate(examples: list[dict[str, torch.Tensor]]) -> SequenceBatch:
        x = torch.stack([e["X"] for e in examples], dim=0)
        a = torch.stack([e["A"] for e in examples], dim=0)
        t = torch.stack([e["T"] for e in examples], dim=0)
        y = torch.stack([e["Y"] for e in examples], dim=0)
        m = torch.stack([e["mask"] for e in examples], dim=0)
        return SequenceBatch(X=x, A=a, T=t, Y=y, mask=m)

    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle) if sampler is None else False,
        sampler=sampler,
        num_workers=0,
        collate_fn=collate,
        drop_last=bool(drop_last),
    )


def _split_indices(n: int, *, test_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    n = int(n)
    if n <= 0:
        return [], []
    if not (0.0 <= float(test_ratio) < 1.0):
        raise ValueError(f"test_ratio must be in [0,1), got {test_ratio}")

    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n, generator=g).tolist()
    n_test = int(n * float(test_ratio))
    if test_ratio > 0.0:
        n_test = max(1, min(n - 1, n_test))
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]
    return train_idx, test_idx


def latest_checkpoint(run_dir: Path, model: str) -> Path | None:
    if model == "timegan":
        p = run_dir / "ckpt_timegan.pt"
        return p if p.exists() else None
    patterns = {
        CAUSALSEQGAN_ID: f"ckpt_{CAUSALSEQGAN_ID}_*_ep*.pt",
        "rcgan": "ckpt_rcgan_*_ep*.pt",
        "vae": "ckpt_vae_ep*.pt",
        "diffusion": "ckpt_diffusion_ep*.pt",
        "crn": "ckpt_crn_ep*.pt",
    }
    files = list(run_dir.glob(patterns[model]))
    if not files:
        return None

    def checkpoint_order(path: Path) -> tuple[int, int]:
        match = re.search(r"(?:_([ABC]))?_ep(\d+)\.pt$", path.name)
        if match is None:
            raise ValueError(f"Invalid checkpoint filename: {path.name}")
        return {None: 0, "A": 1, "B": 2, "C": 3}[match[1]], int(match[2])

    return max(files, key=checkpoint_order)


def _train_adversarial_generator(
    model_name: str,
    *,
    train_dl: DataLoader,
    meta: dict[str, Any],
    device: torch.device,
    out_dir: Path,
    knobs: dict[str, Any],
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    nan_debugger = None
    if knobs.get("debug_nans", False):
        try:
            from src.diagnostics import NaNDebugger

            nan_debugger = NaNDebugger(out_dir / "nan_debug.log")
            print(f"[diag] NaN debugger enabled: {nan_debugger.log_path}")
        except Exception as e:
            print(f"[diag] NaN debugger init failed: {e}")

    epochs_a = int(knobs.get("epochs_a", 2))
    epochs_b = int(knobs.get("epochs_b", 2))
    epochs_c = int(knobs.get("epochs_c", 2))
    lr_g = float(knobs.get("lr_g", 1e-3))
    lr_d = float(knobs.get("lr_d", 1e-4))
    gp_weight = float(knobs.get("gp_weight", 10.0))
    w_y = float(knobs.get("w_y", 1.0))
    w_adv = float(knobs.get("w_adv", 0.5))
    w_sup = float(knobs.get("w_sup", 1.0))
    is_causalseqgan = model_name == CAUSALSEQGAN_ID

    w_do = float(knobs.get("w_do", 0.0))
    w_advT_causal = float(knobs.get("w_advT_causal", knobs.get("w_advT", 0.0)))
    w_t_pred_spurious = float(knobs.get("w_t_pred_spurious", 0.0))
    w_cf = float(knobs.get("w_cf", 0.0))
    grl_lambda = float(knobs.get("grl_lambda", 1.0))
    lr_advT = float(knobs.get("lr_advT", 1e-4))
    save_checkpoints = bool(knobs.get("save_checkpoints", True))
    ckpt_every = max(1, int(knobs.get("ckpt_every", 1)))
    disc_input = str(knobs.get("disc_input", "logits"))
    do_horizon = int(knobs.get("do_horizon", 0))
    do_horizon_train = int(knobs.get("do_horizon_train", do_horizon))
    causal_every = max(1, int(knobs.get("causal_every", 20)))
    cf_every = max(1, int(knobs.get("cf_every", knobs.get("cf_interval", 20))))
    ref_estimator = knobs.get("ref_estimator", None)
    float(knobs.get("do_huber_delta", 1.0))
    max_grad_norm = float(knobs.get("max_grad_norm", 5.0))

    do_time_index = int(knobs.get("do_time_index", 5))
    do_time_sampling = str(knobs.get("do_time_sampling", "random"))
    do_num_time_samples = int(knobs.get("do_num_time_samples", 1))
    do_min_arm_samples = int(knobs.get("do_min_arm_samples", 8))
    do_actions_raw = knobs.get("do_actions", "0,1")
    cf_num_time_samples = int(knobs.get("cf_num_time_samples", 1))

    log_every = int(knobs.get("log_every", 50))
    log_every = max(1, log_every)

    if do_time_sampling not in {"fixed", "random", "all"}:
        raise ValueError(
            f"do_time_sampling must be one of fixed|random|all, got {do_time_sampling}"
        )
    do_num_time_samples = max(1, int(do_num_time_samples))
    do_min_arm_samples = max(1, int(do_min_arm_samples))
    cf_num_time_samples = max(1, int(cf_num_time_samples))

    criterion_huber = nn.L1Loss(reduction="mean")

    if isinstance(do_actions_raw, str):
        do_actions = [int(x.strip()) for x in do_actions_raw.split(",") if x.strip()]
    elif isinstance(do_actions_raw, (list, tuple)):
        do_actions = [int(x) for x in do_actions_raw]
    else:
        raise TypeError("do_actions must be a comma-separated string or a list/tuple of ints")
    if not do_actions:
        do_actions = [0, 1]

    a_emb_dim = int(knobs.get("a_emb_dim", 32))
    t_emb_dim = int(knobs.get("t_emb_dim", 16))
    dropout = float(knobs.get("dropout", 0.1))
    d_disc_h = int(knobs.get("d_disc_h", 128))
    disc_num_layers = int(knobs.get("disc_num_layers", 1))

    a_vocab_size = int(meta["a_vocab_size"] or 100)
    t_vocab_size = int(meta["t_vocab_size"] or 2)

    disc_cfg = SequenceDiscriminatorConfig(
        d_x=int(meta["d_x"]),
        a_is_discrete=bool(meta["a_is_discrete"]),
        a_vocab_size=a_vocab_size,
        a_emb_dim=a_emb_dim,
        d_a=int(meta["d_a"]),
        t_is_discrete=bool(meta["t_is_discrete"]),
        t_vocab_size=t_vocab_size,
        t_emb_dim=t_emb_dim,
        d_t=int(meta["d_t"]),
        d_y=1,
        d_h=d_disc_h,
        num_layers=disc_num_layers,
        dropout=dropout,
    )
    disc = SequenceDiscriminator(disc_cfg).to(device)

    d_eps = int(knobs.get("d_eps", 16))
    if is_causalseqgan:
        gen_cfg = CausalSeqGANConfig(
            d_x=int(meta["d_x"]),
            d_k=int(knobs.get("d_k", 128)),
            k_c_ratio=float(knobs.get("k_c_ratio", 0.5)),
            d_eps=d_eps,
            a_is_discrete=bool(meta["a_is_discrete"]),
            a_vocab_size=a_vocab_size,
            a_emb_dim=a_emb_dim,
            d_a=int(meta["d_a"]),
            t_is_discrete=bool(meta["t_is_discrete"]),
            t_vocab_size=t_vocab_size,
            t_emb_dim=t_emb_dim,
            d_t=int(meta["d_t"]),
            y_dist="bernoulli",
            d_y=int(meta["d_y"]),
            use_y_in_dynamics=bool(knobs.get("y_in_dynamics", False)),
            dynamics=str(knobs.get("dynamics", "gru")),
            mlp_hidden=int(knobs.get("mlp_hidden", 128)),
            dropout=dropout,
            y_head_hidden=int(knobs.get("y_head_hidden", 0)),
            y_head_layers=int(knobs.get("y_head_layers", 1)),
            y_head_mode=str(knobs.get("y_head_mode", "joint")),
            tf_n_layers=int(knobs.get("tf_n_layers", 2)),
            tf_n_heads=int(knobs.get("tf_n_heads", 4)),
            tf_ffn_hidden=int(knobs.get("tf_ffn_hidden", 256)),
            tf_max_seq_len=int(knobs.get("tf_max_seq_len", 512)),
        )
        gen: torch.nn.Module = CausalSeqGAN(gen_cfg).to(device)
    elif model_name == "rcgan":
        gen_cfg = RCGANConfig(
            d_x=int(meta["d_x"]),
            d_eps=d_eps,
            a_is_discrete=bool(meta["a_is_discrete"]),
            a_vocab_size=a_vocab_size,
            a_emb_dim=a_emb_dim,
            d_a=int(meta["d_a"]),
            t_is_discrete=bool(meta["t_is_discrete"]),
            t_vocab_size=t_vocab_size,
            t_emb_dim=t_emb_dim,
            d_t=int(meta["d_t"]),
            rnn=str(knobs.get("rnn", "gru")),
            d_h=int(knobs.get("rcgan_hidden", 128)),
            dropout=dropout,
            use_prev_y=bool(knobs.get("rcgan_use_prev_y", True)),
        )
        gen = RCGANGenerator(gen_cfg).to(device)
    else:
        raise ValueError(f"Unexpected model={model_name}")

    # Treatment classifiers for adversarial deconfounding (K_c) and spurious policy fit (K_s).
    t_clf: Optional[TreatmentClassifier] = None
    t_policy_clf: Optional[TreatmentClassifier] = None
    opt_tclf: Optional[torch.optim.Optimizer] = None
    opt_tpolicy: Optional[torch.optim.Optimizer] = None
    if is_causalseqgan:
        t_clf = getattr(gen, "t_head", None)
        t_policy_clf = getattr(gen, "t_policy_head", None)
        if w_advT_causal > 0.0 and epochs_c > 0 and t_clf is not None:
            opt_tclf = torch.optim.Adam(t_clf.parameters(), lr=lr_advT)
        if w_t_pred_spurious > 0.0 and epochs_c > 0 and t_policy_clf is not None:
            opt_tpolicy = torch.optim.Adam(t_policy_clf.parameters(), lr=lr_advT)

    head_params: list[torch.nn.Parameter] = []
    if t_clf is not None:
        head_params.extend(list(t_clf.parameters()))
    if t_policy_clf is not None:
        head_params.extend(list(t_policy_clf.parameters()))
    head_param_ids = {id(p) for p in head_params}
    gen_params = [p for p in gen.parameters() if id(p) not in head_param_ids]
    opt_g = torch.optim.Adam(
        gen_params,
        lr=lr_g,
        weight_decay=float(knobs.get("weight_decay", 0.0)),
    )
    opt_d = torch.optim.Adam(disc.parameters(), lr=lr_d)

    last_ckpt_payload: Optional[dict[str, Any]] = None

    def save_ckpt(stage: str, epoch: int) -> Path:
        ckpt = {
            "model": model_name,
            "stage": stage,
            "epoch": int(epoch),
            "model_cfg": asdict(gen_cfg),
            "model_state": gen.state_dict(),
            "disc_cfg": asdict(disc_cfg),
            "disc_state": disc.state_dict(),
        }
        path = out_dir / f"ckpt_{model_name}_{stage}_ep{epoch}.pt"
        nonlocal last_ckpt_payload
        last_ckpt_payload = ckpt
        if save_checkpoints and (int(epoch) % ckpt_every == 0):
            torch.save(ckpt, path)
        return path

    last_ckpt = save_ckpt("A", 0)

    # Stage A: supervised pretrain
    if epochs_a > 0:
        print(f"    [Stage A] Supervised pretraining ({epochs_a} epochs)")
    gen.train()
    for ep in range(1, epochs_a + 1):
        total_loss = 0.0
        steps = 0
        for batch in train_dl:
            batch = move_batch(batch, device)
            X, A, T, Y, M = batch.X, batch.A, batch.T, batch.Y, batch.mask

            if is_causalseqgan:
                bsz, seq_len = A.shape[0], A.shape[1]
                eps = torch.zeros(bsz, seq_len, d_eps, device=device)
                out = gen.teacher_forcing(  # type: ignore[attr-defined]
                    x=X, a=A, t=T, y=Y, mask=M, eps=eps, eps_mode="zero"
                )
                loss_y = bernoulli_nll_from_logits(out["y_logits"], Y, mask=M)

                k_tf = out["k"]
                a_enc = gen.encode_a(A.view(-1, *A.shape[2:])).view(bsz, seq_len, -1)  # type: ignore[attr-defined]
                t_enc = gen.encode_t(T.view(-1, *T.shape[2:])).view(bsz, seq_len, -1)  # type: ignore[attr-defined]
                x_rep = gen.x_proj(X.float())[:, None, :].expand(bsz, seq_len, X.shape[1])  # type: ignore[attr-defined]
                if gen_cfg.use_y_in_dynamics:
                    y_dyn = Y.float()
                    if y_dyn.ndim == 2:
                        y_dyn = y_dyn[..., None]
                    inp = torch.cat([a_enc, t_enc, x_rep, y_dyn, eps], dim=-1)
                else:
                    inp = torch.cat([a_enc, t_enc, x_rep, eps], dim=-1)
                k_pred = gen.supervisor(k_tf[:, :-1, :], inp)  # type: ignore[attr-defined]
                loss_sup = (
                    (k_pred - k_tf[:, 1:, :]).pow(2).mean(dim=-1) * M
                ).sum() / M.sum().clamp(min=1.0)

                loss = w_y * loss_y + w_sup * loss_sup
            else:
                out = gen.teacher_forcing(  # type: ignore[attr-defined]
                    x=X, a=A, t=T, y=Y, mask=M, stochastic_y=False
                )
                loss = bernoulli_nll_from_logits(out["y_logits"], Y, mask=M)

            opt_g.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gen_params, max_norm=max_grad_norm)
            opt_g.step()

            total_loss += float(loss.item())
            steps += 1
            if steps % log_every == 0:
                avg = total_loss / max(1, steps)
                print(f"      Stage A | Epoch {ep}/{epochs_a} | Step {steps} | AvgLoss {avg:.4f}")

        avg_loss = total_loss / max(1, steps)
        print(f"      Stage A | Epoch {ep}/{epochs_a} | AvgLoss {avg_loss:.4f}")
        last_ckpt = save_ckpt("A", ep)

    # Stages B and C add adversarial training and causal regularization.
    for stage_name, epochs in [("B", epochs_b), ("C", epochs_c)]:
        if epochs <= 0:
            continue
        print(f"    [Stage {stage_name}] Adversarial training ({epochs} epochs)")
        gen.train()
        disc.train()
        if t_clf is not None:
            t_clf.train()
        if t_policy_clf is not None:
            t_policy_clf.train()

        enable_causal = (
            stage_name == "C"
            and is_causalseqgan
            and (w_do > 0.0 or w_advT_causal > 0.0 or w_t_pred_spurious > 0.0 or w_cf > 0.0)
        )
        global_step = 0
        for ep in range(1, epochs + 1):
            g_loss_accum = 0.0
            d_loss_accum = 0.0
            do_accum = 0.0
            advt_accum = 0.0
            t_pred_accum = 0.0
            cf_accum = 0.0
            steps = 0
            for batch in train_dl:
                batch = move_batch(batch, device)
                X, A, T, Y, M = batch.X, batch.A, batch.T, batch.Y, batch.mask
                bsz, seq_len = A.shape[0], A.shape[1]
                should_causal = global_step % causal_every == 0
                should_cf = global_step % cf_every == 0

                ro = gen.rollout(x=X, a=A, t_obs=T, mask=M, stochastic_y=False)  # type: ignore[attr-defined]
                y_fake = ro["y"]
                y_fake_in = y_fake
                y_real_in = Y
                if disc_input == "logits":
                    y_fake_in = ro.get("y_logits", None)
                    if y_fake_in is None:
                        y_fake_in = torch.log(
                            torch.clamp(y_fake, 0.01, 0.99) / torch.clamp(1.0 - y_fake, 0.01, 0.99)
                        )
                    y_real_in = torch.log(
                        torch.clamp(Y, 0.01, 0.99) / torch.clamp(1.0 - Y, 0.01, 0.99)
                    )
                elif disc_input == "sampled":
                    u = torch.rand_like(y_fake)
                    y_hard = (y_fake > u).float()
                    y_fake_in = y_hard + (y_fake - y_fake.detach())

                real_seq = disc.encode_inputs(x=X, a=A, t=T, y=y_real_in)
                fake_seq = disc.encode_inputs(x=X, a=A, t=T, y=y_fake_in.detach())

                loss_d = wgan_gp_d_loss(disc, real_seq, fake_seq, gp_weight=gp_weight, mask=M)
                opt_d.zero_grad(set_to_none=True)
                loss_d.backward()
                opt_d.step()
                d_loss_accum += float(loss_d.item())

                loss_do = torch.tensor(0.0, device=device)
                loss_advT = torch.tensor(0.0, device=device)
                loss_t_pred = torch.tensor(0.0, device=device)
                loss_cf = torch.tensor(0.0, device=device)

                if is_causalseqgan:
                    eps_seq = torch.randn(bsz, seq_len, d_eps, device=device)
                    tf = gen.teacher_forcing(  # type: ignore[attr-defined]
                        x=X, a=A, t=T, y=Y, mask=M, eps=eps_seq, eps_mode="random"
                    )
                    loss_y = bernoulli_nll_from_logits(tf["y_logits"], Y, mask=M)

                    k_tf = tf["k"]
                    k_c = tf.get("k_c")
                    k_s = tf.get("k_s")
                    if k_c is None or k_s is None:
                        k_c, k_s = gen.split_k(k_tf)
                    a_enc = gen.encode_a(A.view(-1, *A.shape[2:])).view(bsz, seq_len, -1)  # type: ignore[attr-defined]
                    t_enc = gen.encode_t(T.view(-1, *T.shape[2:])).view(bsz, seq_len, -1)  # type: ignore[attr-defined]
                    x_feat = gen.x_proj(X.float())  # type: ignore[attr-defined]
                    x_rep = x_feat[:, None, :].expand(bsz, seq_len, X.shape[1])
                    if gen_cfg.use_y_in_dynamics:
                        y_dyn = Y.float()
                        if y_dyn.ndim == 2:
                            y_dyn = y_dyn[..., None]
                        inp = torch.cat([a_enc, t_enc, x_rep, y_dyn, eps_seq], dim=-1)
                    else:
                        inp = torch.cat([a_enc, t_enc, x_rep, eps_seq], dim=-1)
                    k_pred = gen.supervisor(k_tf[:, :-1, :], inp)  # type: ignore[attr-defined]
                    loss_sup = (
                        (k_pred - k_tf[:, 1:, :]).pow(2).mean(dim=-1) * M
                    ).sum() / M.sum().clamp(min=1.0)
                    if nan_debugger is not None:
                        nan_debugger.check_tensors(
                            stage=str(stage_name),
                            epoch=ep,
                            step=global_step,
                            tag="tf",
                            tensors={
                                "y_logits": tf.get("y_logits"),
                                "k_tf": k_tf,
                                "k_c": k_c,
                                "k_s": k_s,
                                "a_enc": a_enc,
                                "t_enc": t_enc,
                                "x_feat": x_feat,
                                "k_pred": k_pred,
                            },
                        )

                    if enable_causal and bool(gen_cfg.t_is_discrete) and T.ndim == 2:
                        if (
                            w_do > 0.0
                            and should_causal
                            and do_horizon_train > 0
                            and ref_estimator is not None
                        ):
                            support_pairs = []
                            for t_idx in range(seq_len):
                                valid = M[:, int(t_idx)] > 0.5
                                if valid.sum().item() < do_min_arm_samples:
                                    continue
                                for a_val in do_actions:
                                    sel = (T[:, int(t_idx)].long() == int(a_val)) & valid
                                    if sel.sum().item() >= do_min_arm_samples:
                                        support_pairs.append((int(t_idx), int(a_val)))

                            batch_ref = TrajectoryBatch(
                                X=X.detach().cpu(),
                                A=A.detach().cpu(),
                                T=T.detach().cpu(),
                                Y=Y.detach().cpu(),
                                mask=M.detach().cpu(),
                                lengths=compute_lengths(M.detach().cpu()),
                            )
                            ref_ok = False
                            max_ref_tries = int(knobs.get("do_ref_max_tries", 5))
                            t0 = 0
                            action = 0
                            horizon_eff = 0
                            ref_mu = None
                            for _ in range(max_ref_tries):
                                if not support_pairs:
                                    break
                                idx = int(
                                    torch.randint(
                                        low=0, high=len(support_pairs), size=(1,), device=device
                                    ).item()
                                )
                                t0, action = support_pairs[idx]
                                horizon_eff = min(int(do_horizon_train), max(0, seq_len - t0 - 1))
                                ref = ref_estimator.estimate_do(
                                    batch_ref,
                                    t0=int(t0),
                                    horizon=int(horizon_eff),
                                    action=int(action),
                                    subgroup={"name": "all"},
                                    n_boot=1,
                                    seed=int(knobs.get("seed", 0)),
                                )
                                ref_mu = torch.as_tensor(ref["mu"], device=device).float()
                                if (
                                    not bool(ref.get("valid", True))
                                    or not torch.isfinite(ref_mu).all()
                                ):
                                    continue
                                ref_ok = True
                                break

                            if ref_ok and ref_mu is not None:
                                steps = t0 + horizon_eff + 1
                                t_do = torch.full((bsz,), int(action), device=device, dtype=T.dtype)
                                ro_do = gen.rollout(  # type: ignore[attr-defined]
                                    x=X,
                                    a=A,
                                    t_obs=T,
                                    do_t={int(t0): t_do},
                                    mask=M,
                                    eps=eps_seq,
                                    steps=steps,
                                    stochastic_y=False,
                                    causal_only=True,
                                )
                                y_do = ro_do["y"]
                                if y_do.ndim == 3 and y_do.shape[-1] == 1:
                                    y_do = y_do.squeeze(-1)
                                y_slice = y_do[:, int(t0) : steps]
                                m_slice = M[:, int(t0) : steps]
                                gen_mu = (y_slice * m_slice).sum(dim=0) / m_slice.sum(dim=0).clamp(
                                    min=1.0
                                )
                                ref_mu_clipped = torch.clamp(ref_mu, min=0.0, max=1.0)
                                gen_mu_aligned = gen_mu[: ref_mu_clipped.shape[0]]
                                raw_loss_do = criterion_huber(gen_mu_aligned, ref_mu_clipped)
                                if not torch.isfinite(raw_loss_do):
                                    print(
                                        f"      [WARNING] loss_do NaN/Inf at t0={t0}, action={action}; skipping."
                                    )
                                    loss_do = torch.tensor(0.0, device=device)
                                else:
                                    loss_do = raw_loss_do
                                    print(
                                        f"      loss_do_multistep={float(loss_do.item()):.4f} "
                                        f"do_horizon={int(horizon_eff)} t0_sampled={t0}"
                                    )
                        elif w_do > 0.0 and should_causal:
                            # Single-step MMD alignment.
                            if do_time_sampling == "fixed":
                                do_time_indices = [int(do_time_index)]
                            elif do_time_sampling == "random":
                                do_time_indices = torch.randint(
                                    low=0,
                                    high=seq_len,
                                    size=(int(do_num_time_samples),),
                                    device=device,
                                ).tolist()
                            else:
                                do_time_indices = list(range(seq_len))

                            do_terms = 0
                            for t_idx in do_time_indices:
                                if not (0 <= int(t_idx) < int(seq_len)):
                                    continue
                                valid = M[:, int(t_idx)] > 0.5
                                if valid.sum().item() < do_min_arm_samples:
                                    continue

                                k_t = k_tf[:, int(t_idx), :]  # pre-treatment state K_t
                                a_enc_t = gen.encode_a(A[:, int(t_idx)])  # type: ignore[attr-defined]

                                for a_val in do_actions:
                                    a_val = int(a_val)
                                    t_do = torch.full((bsz,), a_val, device=device, dtype=T.dtype)
                                    t_enc_do = gen.encode_t(t_do)  # type: ignore[attr-defined]
                                    y_logits_do = gen.outcome_logits(  # type: ignore[attr-defined]
                                        k_t,
                                        a_enc_t,
                                        t_enc_do,
                                        x_feat,
                                        t_value=t_do,
                                        causal_only=True,
                                    )
                                    y_prob_do = torch.sigmoid(y_logits_do)
                                    feat_do = torch.cat([x_feat, k_t, y_prob_do], dim=-1)[valid]

                                    sel = (T[:, int(t_idx)].long() == a_val) & valid
                                    if sel.sum().item() < do_min_arm_samples:
                                        continue
                                    y_real = Y[sel, int(t_idx)].float().view(-1, 1)
                                    feat_real = torch.cat([x_feat[sel], k_t[sel], y_real], dim=-1)
                                    loss_do = loss_do + mmd_rbf(feat_real, feat_do)
                                    do_terms += 1

                            if do_terms > 0:
                                loss_do = loss_do / float(do_terms)
                            else:
                                loss_do = torch.tensor(0.0, device=device)

                        # (B) adversarial deconfounding (GRL): make K_t less predictive of T_t
                        if (
                            w_advT_causal > 0.0
                            and t_clf is not None
                            and opt_tclf is not None
                            and k_c.shape[-1] > 0
                        ):
                            with torch.no_grad():
                                k_repr_detached = k_c[:, :-1, :].detach()
                            t_logits_clf = t_clf(k_repr_detached)
                            loss_tclf = treatment_ce_loss(t_logits_clf, T.long(), mask=M)
                            opt_tclf.zero_grad(set_to_none=True)
                            loss_tclf.backward()
                            opt_tclf.step()

                            t_logits_adv = t_clf(grad_reverse(k_c[:, :-1, :], lambd=grl_lambda))
                            loss_advT = treatment_ce_loss(t_logits_adv, T.long(), mask=M)

                        if (
                            w_t_pred_spurious > 0.0
                            and t_policy_clf is not None
                            and opt_tpolicy is not None
                            and k_s.shape[-1] > 0
                        ):
                            with torch.no_grad():
                                k_s_detached = k_s[:, :-1, :].detach()
                            t_logits_policy = t_policy_clf(k_s_detached)
                            loss_tpolicy = treatment_ce_loss(t_logits_policy, T.long(), mask=M)
                            opt_tpolicy.zero_grad(set_to_none=True)
                            loss_tpolicy.backward()
                            opt_tpolicy.step()

                            t_logits_policy = t_policy_clf(k_s[:, :-1, :])
                            loss_t_pred = treatment_ce_loss(t_logits_policy, T.long(), mask=M)

                        # Compare the shared state prefix before treatment changes.
                        if w_cf > 0.0 and should_cf:
                            num_actions = max(1, int(t_vocab_size))
                            cf_time_indices = torch.randint(
                                low=0, high=seq_len, size=(int(cf_num_time_samples),), device=device
                            ).tolist()
                            ro_f = gen.rollout(  # type: ignore[attr-defined]
                                x=X, a=A, t_obs=T, mask=M, eps=eps_seq, stochastic_y=False
                            )
                            k_f = ro_f["k"]
                            cf_terms = 0
                            for t_idx in cf_time_indices:
                                if not (0 <= int(t_idx) < int(seq_len)):
                                    continue
                                alt = (T[:, int(t_idx)].long() + 1) % num_actions
                                ro_cf = gen.rollout(  # type: ignore[attr-defined]
                                    x=X,
                                    a=A,
                                    t_obs=T,
                                    do_t={int(t_idx): alt},
                                    mask=M,
                                    eps=eps_seq,
                                    stochastic_y=False,
                                )
                                k_cf = ro_cf["k"]
                                prefix_len = int(t_idx) + 1  # compare K_0..K_t
                                if prefix_len <= 0:
                                    continue
                                k1 = k_f[:, :prefix_len, :]
                                k2 = k_cf[:, :prefix_len, :]
                                mask_k = torch.cat(
                                    [torch.ones(bsz, 1, device=device), M[:, : int(t_idx)]], dim=1
                                )
                                diff = (k1 - k2).pow(2).mean(dim=-1)  # [B, prefix_len]
                                loss_cf = loss_cf + (diff * mask_k).sum() / mask_k.sum().clamp(
                                    min=1.0
                                )
                                cf_terms += 1
                            if cf_terms > 0:
                                loss_cf = loss_cf / float(cf_terms)
                else:
                    tf = gen.teacher_forcing(  # type: ignore[attr-defined]
                        x=X, a=A, t=T, y=Y, mask=M, stochastic_y=False
                    )
                    loss_y = bernoulli_nll_from_logits(tf["y_logits"], Y, mask=M)
                    loss_sup = torch.tensor(0.0, device=device)

                fake_seq_g = disc.encode_inputs(x=X, a=A, t=T, y=y_fake_in)
                loss_adv = wgan_g_loss(disc, fake_seq_g, mask=M)

                loss_g = w_y * loss_y + w_sup * loss_sup + w_adv * loss_adv
                if enable_causal:
                    loss_g = (
                        loss_g
                        + w_do * loss_do
                        + w_advT_causal * loss_advT
                        + w_t_pred_spurious * loss_t_pred
                        + w_cf * loss_cf
                    )
                if nan_debugger is not None:
                    nan_debugger.check_tensors(
                        stage=str(stage_name),
                        epoch=ep,
                        step=global_step,
                        tag="loss",
                        tensors={
                            "loss_y": loss_y,
                            "loss_sup": loss_sup,
                            "loss_adv": loss_adv,
                            "loss_do": loss_do,
                            "loss_advT": loss_advT,
                            "loss_t_pred": loss_t_pred,
                            "loss_cf": loss_cf,
                            "loss_g": loss_g,
                        },
                    )

                opt_g.zero_grad(set_to_none=True)
                loss_g.backward()
                if nan_debugger is not None:
                    nan_debugger.check_grads(gen, stage=str(stage_name), epoch=ep, step=global_step)
                torch.nn.utils.clip_grad_norm_(gen_params, max_norm=max_grad_norm)
                opt_g.step()
                if nan_debugger is not None:
                    nan_debugger.check_params(
                        gen, stage=str(stage_name), epoch=ep, step=global_step
                    )

                g_loss_accum += float(loss_g.item())
                if enable_causal:
                    do_accum += float(loss_do.item())
                    advt_accum += float(loss_advT.item())
                    t_pred_accum += float(loss_t_pred.item())
                    cf_accum += float(loss_cf.item())
                steps += 1
                global_step += 1
                if steps % log_every == 0:
                    avg_d = d_loss_accum / max(1, steps)
                    avg_g = g_loss_accum / max(1, steps)
                    msg = f"      Stage {stage_name} | Epoch {ep}/{epochs} | Step {steps} | D {avg_d:.4f} | G {avg_g:.4f}"
                    if enable_causal:
                        msg += (
                            f" | do {do_accum / max(1, steps):.4f}"
                            f" | advT {advt_accum / max(1, steps):.4f}"
                            f" | t_pred {t_pred_accum / max(1, steps):.4f}"
                            f" | cf {cf_accum / max(1, steps):.4f}"
                        )
                    print(msg)

            avg_d = d_loss_accum / max(1, steps)
            avg_g = g_loss_accum / max(1, steps)
            msg = f"      Stage {stage_name} | Epoch {ep}/{epochs} | D {avg_d:.4f} | G {avg_g:.4f}"
            if enable_causal:
                msg += (
                    f" | do {do_accum / max(1, steps):.4f}"
                    f" | advT {advt_accum / max(1, steps):.4f}"
                    f" | t_pred {t_pred_accum / max(1, steps):.4f}"
                    f" | cf {cf_accum / max(1, steps):.4f}"
                )
            print(msg)
            last_ckpt = save_ckpt(stage_name, ep)

    if not save_checkpoints:
        if last_ckpt_payload is None:
            raise RuntimeError(
                "No checkpoint payload captured; training loop may have not executed."
            )
        torch.save(last_ckpt_payload, last_ckpt)

    return last_ckpt


def _train_vae(
    *,
    train_dl: DataLoader,
    meta: dict[str, Any],
    device: torch.device,
    out_dir: Path,
    knobs: dict[str, Any],
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = int(knobs.get("epochs", 10))
    lr = float(knobs.get("lr", 1e-3))
    kl_weight = float(knobs.get("vae_kl_weight", 0.1))
    dropout = float(knobs.get("dropout", 0.1))
    save_checkpoints = bool(knobs.get("save_checkpoints", True))
    a_emb_dim = int(knobs.get("a_emb_dim", 32))
    t_emb_dim = int(knobs.get("t_emb_dim", 16))

    cfg = SeqVAEConfig(
        d_x=int(meta["d_x"]),
        z_dim=int(knobs.get("vae_z_dim", 32)),
        enc_hidden=int(knobs.get("vae_enc_hidden", 128)),
        dec_hidden=int(knobs.get("vae_dec_hidden", 128)),
        dropout=dropout,
        use_prev_y=True,
        a_is_discrete=bool(meta["a_is_discrete"]),
        a_vocab_size=int(meta["a_vocab_size"] or 100),
        a_emb_dim=a_emb_dim,
        d_a=int(meta["d_a"]),
        t_is_discrete=bool(meta["t_is_discrete"]),
        t_vocab_size=int(meta["t_vocab_size"] or 2),
        t_emb_dim=t_emb_dim,
        d_t=int(meta["d_t"]),
    )
    model = SeqVAE(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    log_every = max(1, int(knobs.get("log_every", 50)))

    last_ckpt_payload: Optional[dict[str, Any]] = None

    def save_ckpt(epoch: int) -> Path:
        ckpt = {
            "model": "vae",
            "epoch": int(epoch),
            "model_cfg": asdict(cfg),
            "model_state": model.state_dict(),
        }
        path = out_dir / f"ckpt_vae_ep{epoch}.pt"
        nonlocal last_ckpt_payload
        last_ckpt_payload = ckpt
        if save_checkpoints:
            torch.save(ckpt, path)
        return path

    last_ckpt = save_ckpt(0)
    if epochs > 0:
        print(f"    [VAE] Training ({epochs} epochs)")
    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0
        for batch in train_dl:
            batch = move_batch(batch, device)
            losses = model.elbo_loss(  # type: ignore[attr-defined]
                x=batch.X, a=batch.A, t=batch.T, y=batch.Y, mask=batch.mask, kl_weight=kl_weight
            )
            opt.zero_grad(set_to_none=True)
            losses["loss"].backward()
            opt.step()
            total_loss += float(losses["loss"].item())
            steps += 1
            if steps % log_every == 0:
                print(
                    f"      VAE | Epoch {ep}/{epochs} | Step {steps} | AvgLoss {total_loss / max(1, steps):.4f}"
                )

        print(f"      VAE | Epoch {ep}/{epochs} | AvgLoss {total_loss / max(1, steps):.4f}")
        last_ckpt = save_ckpt(ep)

    if not save_checkpoints:
        if last_ckpt_payload is None:
            raise RuntimeError(
                "No checkpoint payload captured; training loop may have not executed."
            )
        torch.save(last_ckpt_payload, last_ckpt)

    return last_ckpt


def _train_diffusion(
    *,
    train_dl: DataLoader,
    meta: dict[str, Any],
    device: torch.device,
    out_dir: Path,
    knobs: dict[str, Any],
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = int(knobs.get("epochs", 10))
    lr = float(knobs.get("lr", 1e-3))
    dropout = float(knobs.get("dropout", 0.1))
    save_checkpoints = bool(knobs.get("save_checkpoints", True))
    a_emb_dim = int(knobs.get("a_emb_dim", 32))
    t_emb_dim = int(knobs.get("t_emb_dim", 16))

    cfg = SeqDiffusionConfig(
        d_x=int(meta["d_x"]),
        num_steps=int(knobs.get("diff_steps", 50)),
        beta_start=float(knobs.get("diff_beta_start", 1e-4)),
        beta_end=float(knobs.get("diff_beta_end", 0.02)),
        time_emb_dim=int(knobs.get("diff_time_emb_dim", 64)),
        model_dim=int(knobs.get("diff_model_dim", 128)),
        backbone=str(knobs.get("diff_backbone", "mlp")),
        n_layers=int(knobs.get("diff_n_layers", 2)),
        n_heads=int(knobs.get("diff_n_heads", 4)),
        dropout=dropout,
        a_is_discrete=bool(meta["a_is_discrete"]),
        a_vocab_size=int(meta["a_vocab_size"] or 100),
        a_emb_dim=a_emb_dim,
        d_a=int(meta["d_a"]),
        t_is_discrete=bool(meta["t_is_discrete"]),
        t_vocab_size=int(meta["t_vocab_size"] or 2),
        t_emb_dim=t_emb_dim,
        d_t=int(meta["d_t"]),
        y_dim=1,
    )
    model = SeqDiffusion(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    log_every = max(1, int(knobs.get("log_every", 50)))

    last_ckpt_payload: Optional[dict[str, Any]] = None

    def save_ckpt(epoch: int) -> Path:
        ckpt = {
            "model": "diffusion",
            "epoch": int(epoch),
            "model_cfg": asdict(cfg),
            "model_state": model.state_dict(),
        }
        path = out_dir / f"ckpt_diffusion_ep{epoch}.pt"
        nonlocal last_ckpt_payload
        last_ckpt_payload = ckpt
        if save_checkpoints:
            torch.save(ckpt, path)
        return path

    last_ckpt = save_ckpt(0)
    if epochs > 0:
        print(f"    [Diffusion] Training ({epochs} epochs)")
    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0
        for batch in train_dl:
            batch = move_batch(batch, device)
            loss = model.denoising_loss(  # type: ignore[attr-defined]
                x=batch.X, a=batch.A, t=batch.T, y=batch.Y, mask=batch.mask
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
            steps += 1
            if steps % log_every == 0:
                print(
                    f"      Diffusion | Epoch {ep}/{epochs} | Step {steps} | AvgLoss {total_loss / max(1, steps):.4f}"
                )

        print(f"      Diffusion | Epoch {ep}/{epochs} | AvgLoss {total_loss / max(1, steps):.4f}")
        last_ckpt = save_ckpt(ep)

    if not save_checkpoints:
        if last_ckpt_payload is None:
            raise RuntimeError(
                "No checkpoint payload captured; training loop may have not executed."
            )
        torch.save(last_ckpt_payload, last_ckpt)

    return last_ckpt


def _train_crn(
    *,
    train_dl: DataLoader,
    meta: dict[str, Any],
    device: torch.device,
    out_dir: Path,
    knobs: dict[str, Any],
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = int(knobs.get("epochs", 10))
    lr = float(knobs.get("lr", 1e-3))
    dropout = float(knobs.get("dropout", 0.1))
    save_checkpoints = bool(knobs.get("save_checkpoints", True))
    a_emb_dim = int(knobs.get("a_emb_dim", 32))
    t_emb_dim = int(knobs.get("t_emb_dim", 16))
    w_treat = float(knobs.get("crn_w_treat", 1.0))

    cfg = CRNConfig(
        d_x=int(meta["d_x"]),
        d_h=int(knobs.get("crn_hidden", 128)),
        dropout=dropout,
        grl_lambda=float(knobs.get("crn_grl_lambda", 1.0)),
        a_is_discrete=bool(meta["a_is_discrete"]),
        a_vocab_size=int(meta["a_vocab_size"] or 100),
        a_emb_dim=a_emb_dim,
        d_a=int(meta["d_a"]),
        t_is_discrete=bool(meta["t_is_discrete"]),
        t_vocab_size=int(meta["t_vocab_size"] or 2),
        t_emb_dim=t_emb_dim,
        d_t=int(meta["d_t"]),
    )
    model = CRN(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    log_every = max(1, int(knobs.get("log_every", 50)))

    last_ckpt_payload: Optional[dict[str, Any]] = None

    def save_ckpt(epoch: int) -> Path:
        ckpt = {
            "model": "crn",
            "epoch": int(epoch),
            "model_cfg": asdict(cfg),
            "model_state": model.state_dict(),
        }
        path = out_dir / f"ckpt_crn_ep{epoch}.pt"
        nonlocal last_ckpt_payload
        last_ckpt_payload = ckpt
        if save_checkpoints:
            torch.save(ckpt, path)
        return path

    last_ckpt = save_ckpt(0)
    if epochs > 0:
        print(f"    [CRN] Training ({epochs} epochs)")
    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0
        for batch in train_dl:
            batch = move_batch(batch, device)
            losses = model.loss(  # type: ignore[attr-defined]
                x=batch.X, a=batch.A, t=batch.T, y=batch.Y, mask=batch.mask, w_treat=w_treat
            )
            opt.zero_grad(set_to_none=True)
            losses["loss"].backward()
            opt.step()
            total_loss += float(losses["loss"].item())
            steps += 1
            if steps % log_every == 0:
                print(
                    f"      CRN | Epoch {ep}/{epochs} | Step {steps} | AvgLoss {total_loss / max(1, steps):.4f}"
                )

        print(f"      CRN | Epoch {ep}/{epochs} | AvgLoss {total_loss / max(1, steps):.4f}")
        last_ckpt = save_ckpt(ep)

    if not save_checkpoints:
        if last_ckpt_payload is None:
            raise RuntimeError(
                "No checkpoint payload captured; training loop may have not executed."
            )
        torch.save(last_ckpt_payload, last_ckpt)

    return last_ckpt


def _train_timegan(
    *,
    train_dl: DataLoader,
    meta: dict[str, Any],
    device: torch.device,
    out_dir: Path,
    knobs: dict[str, Any],
) -> Path:
    from src.model.timegan import TimeGANTrainConfig, fit_timegan  # noqa: WPS433

    out_dir.mkdir(parents=True, exist_ok=True)

    epochs_embed = int(knobs.get("epochs_embed", 3))
    epochs_supervisor = int(knobs.get("epochs_supervisor", 3))
    epochs_joint = int(knobs.get("epochs_joint", 5))

    a_emb_dim = int(knobs.get("a_emb_dim", 32))
    t_emb_dim = int(knobs.get("t_emb_dim", 16))
    x_emb_dim = int(knobs.get("x_emb_dim", 32))
    dropout = float(knobs.get("dropout", 0.1))

    a_vocab_size = int(meta["a_vocab_size"] or 100)
    t_vocab_size = int(meta["t_vocab_size"] or 2)

    cfg = TimeGANConfig(
        d_x=int(meta["d_x"]),
        seq_len=int(meta["seq_len"]),
        a_is_discrete=bool(meta["a_is_discrete"]),
        a_vocab_size=a_vocab_size,
        a_emb_dim=a_emb_dim,
        d_a=int(meta["d_a"]),
        t_is_discrete=bool(meta["t_is_discrete"]),
        t_vocab_size=t_vocab_size,
        t_emb_dim=t_emb_dim,
        d_t=int(meta["d_t"]),
        d_y=int(meta["d_y"]),
        x_emb_dim=x_emb_dim,
        hidden_dim=int(knobs.get("timegan_hidden", 64)),
        num_layers=int(knobs.get("timegan_num_layers", 2)),
        z_dim=int(knobs.get("timegan_z_dim", 16)),
        dropout=dropout,
    )
    model = TimeGAN(cfg).to(device)

    train_cfg = TimeGANTrainConfig(
        epochs_embed=epochs_embed,
        epochs_supervisor=epochs_supervisor,
        epochs_joint=epochs_joint,
        lr_embed=float(knobs.get("lr_embed", 1e-3)),
        lr_gen=float(knobs.get("lr_gen", 1e-3)),
        lr_disc=float(knobs.get("lr_disc", 1e-3)),
        lambda_sup=float(knobs.get("lambda_sup", 10.0)),
        lambda_mom=float(knobs.get("lambda_mom", 1.0)),
        g_steps=int(knobs.get("g_steps", 1)),
        d_steps=int(knobs.get("d_steps", 1)),
        log_every=int(knobs.get("log_every", 50)),
        max_batches_per_epoch=knobs.get("max_batches_per_epoch", 200),
        reservoir_size=int(knobs.get("reservoir_size", 2048)),
    )

    print(
        f"    [TimeGAN] embed={epochs_embed} sup={epochs_supervisor} joint={epochs_joint} "
        f"hidden={cfg.hidden_dim} z={cfg.z_dim} layers={cfg.num_layers}"
    )
    fit_info = fit_timegan(model, train_dl, device=device, train_cfg=train_cfg)

    ckpt = {
        "model": "timegan",
        "epoch": int(epochs_joint),
        "model_cfg": asdict(cfg),
        "model_state": model.state_dict(),
        "train_cfg": asdict(train_cfg),
        "train_history": fit_info.get("history", {}),
    }
    if getattr(model, "_reservoir", None) is not None:
        ckpt["reservoir"] = {k: v.detach().cpu() for k, v in model._reservoir.items()}  # type: ignore[attr-defined]
    if getattr(model, "_x_mean", None) is not None:
        ckpt["x_mean"] = model._x_mean.detach().cpu()  # type: ignore[attr-defined]
    if getattr(model, "_x_std", None) is not None:
        ckpt["x_std"] = model._x_std.detach().cpu()  # type: ignore[attr-defined]

    ckpt_path = out_dir / "ckpt_timegan.pt"
    torch.save(ckpt, ckpt_path)

    try:
        n_gen = int(knobs.get("gen_samples", 512))
        sample = model.generate(n_gen, max_len=int(meta["seq_len"]), conditions=None)
        sample_path = out_dir / "samples_timegan.npz"
        np.savez_compressed(
            sample_path,
            X=sample["X"].numpy(),
            A=sample["A"].numpy(),
            T=sample["T"].numpy(),
            Y=sample["Y"].numpy(),
            M=sample["mask"].numpy(),
        )
        print(f"    [TimeGAN] saved samples: {sample_path.as_posix()}")
    except Exception as e:
        print(f"    [TimeGAN] sample generation failed: {e}")

    return ckpt_path
