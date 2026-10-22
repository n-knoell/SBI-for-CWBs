# SBI: neural posterior estimation

Every script selects a free GPU (`autocvd`) and resolves its paths relative to
this folder, so it can be run from anywhere, e.g. `python SBI/train_posterior.py`.

## Layout

```
SBI/
├── cwb_sbi/                     library
│   ├── simulator.py             hydro simulation -> clean Hα photon counts
│   ├── halpha.py                emissivity, projection, photon counts, detector noise
│   ├── noise.py                 fresh detector noise for training
│   ├── priors.py                BoxUniform prior (bounds shared with ABC)
│   ├── embedding_nets.py        CNN embedding networks
│   ├── posterior_io.py          load / save pickled posteriors
│   ├── data.py                  load simulation files (drops the T = 0 snapshot)
│   ├── diagnostics.py           robust test-set log-probabilities
│   ├── plotting.py              image grids, posterior predictive grid
│   ├── abc_shared.py            units, prior ranges, corner plot from ABC/abc_inference
│   └── paths.py                 folder locations
├── run_simulations.py           1. simulate the training bank (+ reference observations)
├── optimize_hyperparameters.py  2. optuna search (NLL vs TARP)
├── plot_optuna_study.py            plot the optuna study
├── train_posterior.py           3. train the NPE posterior with noise resampling
├── calibrate_posterior.py       4. predicted vs true, PP, PIT, TARP, SBC
├── calibrate_training_size.py      posterior width / error vs training-set size
├── plot_posteriors.py           5. corner plots for the reference observations
├── data/                        simulation banks, test sets, mlr0_array.npy, reference_observations/
├── models/                      trained posteriors (*.pkl)
├── results/                     optuna study database, calibration metrics
└── figures/                     posteriors/, calibration/, optuna/, reference_observations/
```

## Data files not in the repository

The large simulation files are git-ignored. The scripts expect them in `data/`:

| file | used by |
| --- | --- |
| `simulations_clean_74.pt` | `train_posterior.py` |
| `full_comb.pt`, `test_data_full.pt` | `optimize_hyperparameters.py` |
| `test_data_960_full.pt` | `calibrate_posterior.py`, `calibrate_training_size.py`, `plot_posteriors.py` (test-set mode) |

`calibrate_training_size.py` additionally expects
`models/trained_posterior_adv_<N>_trial20.pkl` for every training-set size N.
