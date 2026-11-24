"""Dataset locations and default training settings."""

DATASET_PATHS = {
    "assist09": "data/assist09.npz",
    "assist09_next": "data/assist09_hint_next.npz",
    "oulad": "data/oulad.npz",
    "statics": "data/statics.npz",
    "irt_synth": "data/dkt_synth.npz",
}

ACTIVE_DATASETS = ["irt_synth"]

CAUSALSEQGAN_ID = "scm_causal"
CAUSALSEQGAN_NAME = "CausalSeqGAN"

COMMON_KNOBS = {
    "batch_size": 512,
    "seq_len": 30,
    "device": "auto",
    "seed": 42,
    "test_ratio": 0.2,
    "keep_checkpoints": True,
}

DATASET_SPECIFIC_KNOBS = {
    "oulad": {
        "dynamics": "transformer",
        "w_advT_causal": 0.02,
        "do_horizon_train": 2,
        "batch_size": 1024,
        "d_k": 64,
        "rcgan_hidden": 64,
        "w_do": 0.3,
    },
    "assist09": {
        "dynamics": "transformer",
        "batch_size": 256,
        "w_advT_causal": 0.02,
        "do_horizon_train": 2,
        "d_k": 64,
        "w_do": 0.3,
    },
    "assist09_next": {
        "dynamics": "transformer",
        "batch_size": 256,
        "w_advT_causal": 0.02,
        "w_t_pred_spurious": 0.1,
        "do_horizon_train": 1,
        "causal_every": 4,
        "cf_every": 8,
        "d_k": 64,
        "w_do": 0.05,
        "w_cf": 0.005,
    },
    "statics": {
        "dynamics": "gru",
        "batch_size": 64,
        "d_k": 32,
        "do_horizon_train": 2,
        "w_advT_causal": 0.02,
        "w_do": 0.3,
    },
    "irt_synth": {
        "dynamics": "gru",
        "d_k": 64,
        "disc_input": "sampled",
        "causal_every": 4,
        "cf_every": 8,
        "k_c_ratio": 0.5,
        "w_advT_causal": 0.05,
        "w_y": 1,
        "w_do": 0.8,
        "do_horizon_train": 1,
        "w_cf": 0.0,
        "epochs_c": 20,
    },
}

MODEL_KNOBS = {
    CAUSALSEQGAN_ID: {
        "epochs_a": 30,
        "epochs_b": 30,
        "epochs_c": 20,
        "dynamics": "gru",
        "d_k": 64,
        "k_c_ratio": 0.5,
        "w_do": 0.05,
        "w_cf": 0.02,
        "causal_every": 20,
        "cf_every": 10,
        "grl_lambda": 1.0,
        "lr_advT": 0.0001,
        "do_time_sampling": "random",
        "do_num_time_samples": 2,
        "do_min_arm_samples": 16,
        "do_actions": "0,1",
        "cf_num_time_samples": 1,
        "w_advT_causal": 0.02,
        "w_t_pred_spurious": 0.1,
        "do_horizon_train": 1,
    },
    "rcgan": {"epochs_a": 30, "epochs_b": 30, "epochs_c": 10, "rnn": "gru", "rcgan_hidden": 64},
    "crn": {"epochs": 30, "crn_hidden": 32, "crn_grl_lambda": 1.0},
    "diffusion": {"epochs": 30, "diff_steps": 50, "diff_backbone": "transformer"},
    "vae": {"epochs": 30, "vae_z_dim": 32},
    "timegan": {
        "epochs_embed": 3,
        "epochs_supervisor": 3,
        "epochs_joint": 5,
        "timegan_hidden": 32,
        "timegan_num_layers": 2,
        "timegan_z_dim": 16,
        "x_emb_dim": 32,
        "lambda_sup": 1.0,
        "lambda_mom": 1.0,
        "lr_embed": 0.001,
        "lr_gen": 0.001,
        "lr_disc": 0.001,
        "max_batches_per_epoch": 200,
        "reservoir_size": 2048,
        "gen_samples": 512,
    },
}
