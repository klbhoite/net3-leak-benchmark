# net3-leak-benchmark

Leak detection benchmark on the EPANET Net3 network: code, data and results

This repository reproduces every number and figure in the manuscript
"Hydraulically informed deep and ensemble learning for leak detection in pipeline
networks with storage: a scenario-grouped benchmark".

## Contents

| Folder | What is inside |
|---|---|
| `code/` | Scenario generator, models, cross-validation, analysis and figure scripts |
| `data/` | `net3_leak.npz` (main dataset, 700 scenarios) and `net3_smallleak.npz` (400-scenario stress set), each with a `_meta.json` that lists every scenario's seed, leak node, size, type and onset |
| `results/` | Out-of-fold predictions (`oof_*.npz`), metric tables (`table_*.csv`), training logs, timing, stress tests |
| `figures/` | All figures as 600-dpi PNG and vector PDF |

## Reproducing

Python 3.11+ (tested on 3.13), on a CPU only:

```bash
pip install wntr==1.* numpy pandas scikit-learn xgboost lightgbm torch matplotlib
cd code
python generate_dataset.py --out ../data/net3_leak.npz                              # ~10 min on 4 cores
python generate_dataset.py --dmin 0.008 --dmax 0.05 --seed 3031 --n_leak 300 --n_free 100 --out ../data/net3_smallleak.npz
python run_experiments.py --rep hyd     # proposed input, 5 folds, ~80 min on 4 cores
python run_experiments.py --rep raw     # ablation, ~80 min
python analyze.py                       # tables + figures 4–7
python fig_dataset.py                   # figures 1–3
python run_stress.py                    # noise and small-leak tests
```

All random seeds are fixed (dataset seed 2026, split seed 7). Deep-learning results can
differ in the third decimal place across machines and library versions, because CPU
floating-point operations are not bit-identical.

## Main design choices

* **Splits by scenario.** No window from a test scenario is ever seen in training.
  Normalisation statistics come from the training scenarios only.
* **Labels.** A window is labelled as a leak if a leak is active at its last sample.
* **Ensemble.** `code/ensemble.json` fixes the ensemble members (LogReg, CNN-BiLSTM,
  TCN, Transformer). They were chosen before the test results were inspected.
* **Simulated data.** The data are synthetic. See the Limitations section of the paper.

## Data size and regeneration

The two datasets (`data/net3_leak.npz`, about 27 MB, and `data/net3_smallleak.npz`) are included.
If you prefer to rebuild them, the two `generate_dataset.py` commands above recreate them
from fixed seeds. Trained model weights (`*.pt`) are not included; `run_experiments.py`
retrains all models.

## Licence

Code: MIT (see `LICENSE`). Data and results: CC BY 4.0.
