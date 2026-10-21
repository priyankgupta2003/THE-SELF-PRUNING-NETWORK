# Falsifiable claim

**Claim.** On sklearn digits with a 64-128-64-10 ReLU MLP trained for 60 epochs with the cubic
schedule in this repo, at **98% weight sparsity** (340 of 17,024 weights kept), Fisher/OBD
self-pruning reaches **92.7%** mean test accuracy versus **74.3%** for magnitude pruning
under the identical layerwise budget, schedule, init and batch order: a paired gain of
**+18.4 pp** (95% bootstrap CI [+14.6, +21.8]; wins/ties/losses 10/0/0;
exact sign-flip p = 0.0020, the smallest possible with 10 seeds), on fresh seeds 10-19.
Dense reference on the same seeds: 97.1%.

**What would falsify it.** Running `python experiments/confirm.py` with any other 10 seeds and
getting a 95% CI for the paired difference that includes 0.

**What it does NOT say.** At 90% sparsity Fisher is NOT better: magnitude is, if anything,
slightly ahead (-0.28 pp on these seeds; the bootstrap CI just excludes 0, the exact test gives
p = 0.125; on seeds 0-9 the gap was -0.08 pp, p = 0.82). Fisher's advantage appears only once the budget is tight enough that *which*
weights survive matters more than the network's slack. Also, global magnitude pruning
(results/paired_tests.csv) is within noise of layerwise Fisher up to 98%; Fisher's clear win
over every baseline is at 99%. Part of what Fisher buys is a better per-layer allocation.

## Confirmation set (seeds 10-19, never used during development)

| sparsity | Fisher mean ± sd | magnitude mean ± sd | Fisher − mag (pp) | 95% CI | W/T/L | p |
|---|---|---|---|---|---|---|
| 90% | 96.43% ± 0.34% | 96.71% ± 0.54% | -0.28 | [-0.56, -0.03] | 2/3/5 | 0.1250 |
| 95% | 95.74% ± 0.46% | 94.76% ± 1.22% | +0.97 | [+0.22, +1.84] | 7/0/3 | 0.0527 |
| 98% | 92.70% ± 1.51% | 74.35% ± 5.76% | +18.36 | [+14.57, +21.81] | 10/0/0 | 0.0020 |

Test set: 359 images, 1 image = 0.28 pp. Final-epoch accuracy only; nothing selected on test.

Main-sweep paired tests (seeds 0-9) for every method pair: `results/paired_tests.csv`.
