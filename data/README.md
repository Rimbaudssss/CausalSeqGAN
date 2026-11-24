# Data preparation

Data files are local inputs and are excluded from the code distribution.
Acquire each source under its provider's terms. The repository's MIT license
covers its code, not the source datasets.

## Trajectory datasets

The named datasets in `src/config.py` use these paths:

| Dataset key | Local file |
| --- | --- |
| `irt_synth` | `data/dkt_synth.npz` |
| `assist09` | `data/assist09.npz` |
| `assist09_next` | `data/assist09_hint_next.npz` |
| `oulad` | `data/oulad.npz` |
| `statics` | `data/statics.npz` |

Each archive contains `X`, `A`, `T`, and `Y`, with an optional valid-step mask
`M`. `X` is `[N, d_x]`; sequence arrays share `[N, L]` as their first two
dimensions. Integer contexts and treatments use `[N, L]`, while continuous
inputs use `[N, L, d]`. `Y` contains binary outcomes. `M` is `[N, L]` with
ones for valid steps. Context and treatment indices must be nonnegative.

Use `--data-npz path/to/dataset.npz` for a custom dataset. The original
raw-event preprocessing pipelines for the observational datasets are not part
of this code release; those named datasets require prepared archives conforming
to this interface.

### DKT-Synth

The evaluation additionally requires these oracle arrays:

- `Y_cf`: `[N, L, 2, H+1]`, probabilities after a single-time treatment change.
- `Y_policy`: `[N, 4, L]`, probabilities under the four fixed policy paths.
- `policy_names`: the corresponding policy names.

The reference dataset has SHA-256:

```text
f42321616e26f02a8c9637f02ea08e38ef5bf1f82539413f561e710fd3c49652
```

`checksums.json` records file identities. The benchmark evaluates the first
30 outcomes of the reference dataset; the oracle policies retain their switch
at step 10.

To prepare a new dataset from processed ASSIST09 inputs, train an oracle and
generate trajectories with its checkpoint:

```sh
python scripts/train_dkt_oracle.py --data_npz data/assist09.npz --out checkpoints/dkt_oracle.pt --seed 0 --device cuda
python scripts/make_dkt_synth.py --oracle_path checkpoints/dkt_oracle.pt --out data/dkt_generated.npz --n 20000 --seq_len 50 --with_oracle --oracle_use_cfg --hmax 29 --cf_t0_list 0,5,10 --seed 0 --device cuda
```

This creates a new dataset. Exact reference reproduction requires the matching
oracle weights, data checksum, training settings, and generator checkpoints.

## Randomised benchmark

`src/data_rct.py` reads these files:

| Source | Required path relative to `data/rct/` |
| --- | --- |
| ASSISTments RCT88/89 | `assistments_rct88/processed_experiment_data.csv` |
| ASSISTments LAS2016 | `assistments_las2016/ThisOne.csv` |
| ASSISTments Study2 | `assistments_abtest_study2/Study2 11 New Experiments/TreatmentAssignment_Covariate_Outcome_Study2.csv` |

The RCT88/89 inputs derive from
[Exploring Common Trends in Online Educational Experiments](https://doi.org/10.17605/osf.io/59shv)
and its [expansion](https://doi.org/10.17605/osf.io/m2jqe).
The source project's documentation links the
[ASSISTments data license](https://osf.io/s7ufj).
LAS2016 and Study2 require their respective original ASSISTments distributions;
this repository does not provide replacement download files.

The loaders preserve participant IDs for disjoint splitting, treatment
assignments, outcomes, experiment identifiers, and available pre-treatment
covariates. See their required-column checks in `src/data_rct.py` before
substituting a source file. The fixed benchmark retains 118 binary treatment
tasks from 76 experiments across the three sources.

Fitted assets are expected under `checkpoints/rct/`. The complete saved
baseline prediction matrix is a separate local input at
`results/rct/fold_predictions.csv`; the public aggregate reference CSVs in
`benchmarks/rct/` do not contain participant predictions.
