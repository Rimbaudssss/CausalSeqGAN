# CausalSeqGAN

CausalSeqGAN models treatment-conditioned student outcome trajectories with a
structural generator and an explicit intervention and policy rollout interface.
The randomised benchmark additionally fits a population-effect head using
training participants. `CausalSeqGAN` is the default model for training and
prediction.

The generator predicts outcomes along supplied contexts or items; it does not
generate future contexts. Longitudinal evaluation uses the generator directly.
The fitted effect head is used for the single-step randomised benchmark.

## Installation

Use Python 3.10 or later. From the repository root, create a virtual environment:

```sh
python -m venv .venv
```

Activate it with `.venv\Scripts\Activate.ps1` in PowerShell or
`source .venv/bin/activate` in Bash, then install:

```sh
python -m pip install -e .
python main.py --help
```

Install a CUDA-enabled PyTorch build when GPU training is needed; CPU execution is supported. The
automated checks use Python 3.11 on CPU.

## Quick start

Generate a small IRT dataset and run one supervised training epoch:

```sh
python scripts/make_irt_semi_synth.py --out data/irt_demo.npz --n 256 --seq_len 12 --hmax 11 --seed 42
python main.py --data-npz data/irt_demo.npz --seq_len 12 --epochs_a 1 --epochs_b 0 --epochs_c 0 --batch_size 32 --device cpu --no-eval_tstr --no-eval_privacy --no-eval_calibration --no_plot_cf
```

This example checks the training pipeline; it does not reproduce the DKT-Synth
benchmark. Full training settings are defined in `src/config.py` and can be
overridden through the command line. `causalseqgan` is also available as a console
command after installation.

## Data

Trajectory datasets are NumPy archives with these arrays:

| Array | Shape | Meaning |
| --- | --- | --- |
| `X` | `[N, d_x]` | Static, pre-treatment covariates |
| `A` | `[N, L]` or `[N, L, d_a]` | Item or context sequence |
| `T` | `[N, L]` or `[N, L, d_t]` | Treatment sequence |
| `Y` | `[N, L]` or `[N, L, d_y]` | Binary outcomes |
| `M` | `[N, L]` | Valid-step mask; defaults to ones when absent |

Oracle evaluations additionally require `Y_cf`, `Y_policy`, and `policy_names`.
See [data/README.md](data/README.md) for dataset locations and preparation, and
`data/checksums.json` for the benchmark file identities. Data, fitted weights,
and generated results are local assets and are excluded from Git.

`scripts/train_dkt_oracle.py` trains the DKT oracle;
`scripts/make_dkt_synth.py` generates a dataset from its checkpoint. Reproducing
the reported benchmark requires the matching data and checkpoint files, rather
than a newly generated dataset with different oracle weights.

## Evaluation

### DKT-Synth

Evaluate saved checkpoints at multiple lengths without retraining:

```sh
python scripts/evaluate_multistep.py --data data/dkt_synth.npz --checkpoint-dir checkpoints/dkt --output results/dkt --device cpu
python scripts/build_dkt_table.py --results results/dkt
```

The evaluation reports PEHE, ATE error, policy-value error, and regret for
`K = 1, 5, 10, 20, 30` outcomes. Effect errors use each learner's time-averaged
effect after changing a single treatment, with later factual actions fixed.
These are distinct from terminal-outcome errors. Policies follow fixed
Never Treat, Always Treat, Early-on, and Late-on paths. Their switch remains at
step 10 when shorter prefixes are evaluated; for `K <= 10`, only two prefixes
are distinct. The RCT effect head is not applied.

The sequence comparators are CRN, TimeGAN, RCGAN, VAE, and Diffusion. Their
checkpoint names and participant splits are recorded in evaluation manifests.

### Randomised benchmark

Run the default CausalSeqGAN predictor with its fitted effect estimator:

```sh
python main.py rct
```

Refit both components using the fixed participant folds or summarise an existing
prediction matrix:

```sh
python main.py rct --fit --device cuda
python main.py rct --summarize
```

The benchmark configuration is `configs/rct.json`. Local fitted assets are read
from `checkpoints/rct/`; outputs are written to `results/rct/`. Summary-only
evaluation requires `results/rct/fold_predictions.csv`, including the saved
baseline predictions.

The 118 tasks belong to 76 experiments and six intervention families. Overall
metrics average experiments within sources before averaging sources equally;
RMSE takes the square root within each source. Model seeds share the fixed
participant folds. Their SD measures variation between model runs, not
uncertainty over independently redrawn participant splits.

The effect head is fitted on training data and forms part of the complete RCT
estimator. Its comparison with an uncalibrated baseline does not isolate the
contribution of the neural architecture. RCT policy metrics concern choosing
between treatment arms; they do not establish longitudinal policy quality.

## Python API

```python
from src.model import CausalSeqGAN, CausalSeqGANConfig

model = CausalSeqGAN(CausalSeqGANConfig(d_x=8, a_vocab_size=50, t_vocab_size=2))
```

The model supports factual and intervention rollouts, fitted effect prediction,
and checkpoint save/load. See the configuration and method signatures in
`src/model/causalseqgan.py` and `src/model/effect_head.py`.

## Repository layout

```text
src/          model, training, data, baselines, and evaluation
scripts/      data preparation and benchmark commands
configs/      fixed benchmark configuration
data/         dataset instructions and checksums; local data are ignored
checkpoints/  local fitted weights and effect-head states
results/      generated predictions, metrics, and tables
tests/        model and evaluation checks
```

Run the checks with:

```sh
python -m unittest discover -s tests -v
```

## License

The code is distributed under the [MIT License](LICENSE). Dataset access,
redistribution, and use remain subject to the respective data providers' terms.
