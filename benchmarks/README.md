# Reference results

These CSVs contain aggregate reference metrics and metrics by training seed.
Participant records, raw data, fitted weights, and participant-level predictions
are excluded.

## DKT-Synth

- `dkt/summary.csv`: mean and sample SD for each method, intervention time,
  and evaluation length.
- `dkt/by_seed.csv`: the corresponding seed-level effect and policy metrics.
- `dkt/protocol.json`: evaluation settings and the reference data checksum.

Six methods are evaluated using seeds 42, 43, and 44, with 4,000 held-out
learners per seed and 20 Monte Carlo draws. Evaluation lengths are
1, 5, 10, 20, and 30 outcomes, subject to the remaining sequence length after
the intervention. PEHE and ATE error use each learner's time-averaged effects.
Policy metrics start at intervention time zero; their absent entries at later
intervention times are intentional.

Never Treat, Always Treat, Early-on, and Late-on use fixed paths with the
switch at step 10. At lengths up to 10, they have only two distinct prefixes.
This evaluation uses trajectory generators without the RCT effect head.

## Randomised benchmark

- `rct/overall.csv`: overall means and sample SDs across model seeds.
- `rct/by_seed.csv`: overall metrics for each model seed.
- `rct/families.csv`: results for the six intervention families.

The benchmark includes 118 tasks from 76 experiments and three sources.
Participant folds are fixed across methods and model seeds. Overall metrics
average tasks within experiments, experiments within sources, and sources
equally; RMSE takes the square root within each source. Family summaries use
their own experiment-balanced aggregation, so their simple average need not
equal the overall result. Seed SDs describe variation between model runs.

## Reproduction

Matching these outputs requires matching input data, weights, configuration,
splits, and evaluation settings. The CSVs serve as reference outputs; a new
training run is not guaranteed to reproduce them exactly. Follow the commands
in the root README and the preparation instructions in `data/README.md`.
New outputs are written to `results/`.
