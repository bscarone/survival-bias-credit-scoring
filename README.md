# The Illusion of Improvement: Reject Inference Strategies in Credit Scoring

Code for the ECML PKDD 2026 paper:
> Bruno Scarone, Ricardo Baeza-Yates. *The Illusion of Improvement: Reject Inference Strategies in Credit Scoring*. ECML PKDD 2026. <https://arxiv.org/abs/2606.18479>

## Overview

We simulate iterative model-based lending, in which a credit scoring model is retrained on data shaped by its own approval decisions, and study what survival bias does to it over successive retraining cycles. Specifically, in this work we:

- Uncover a **structural failure mode**: accuracy improves while recall collapses, creating an illusion of improvement, and show that standard metrics systematically reward strategies that amplify survival bias
- Formalize the **accuracy–recall paradox**, proving that the contribution of recall to accuracy vanishes as the training set default rate deflates
- Evaluate **five reject inference strategies** (Parceling, Fuzzy Augmentation, Simple Extrapolation, Twins, Shallow Self-Learning) against a Biased and an Oracle baseline across three datasets and two classifiers
- Propose **controlled exploration** as an assumption-free alternative, in which the lender deliberately approves a fraction of rejected applicants and observes their true outcomes
- Show that exploration rates as low as **2–5%** diagnose the severity of the feedback loop at near-zero cost, and propose to use the training set default rate as a diagnostic the model cannot influence

## Requirements

- Python 3.9 (results produced on 3.9.21)
- See `requirements.txt` for Python dependencies

To install dependencies:

```bash
pip install -r requirements.txt
```

Versions are pinned to those used to produce the published results. `scikit-learn` does not guarantee identical models across releases for a fixed `random_state`, so a different version will give numbers close to but not identical to the paper's.

## Datasets

The paper uses three publicly available credit scoring datasets:

| Dataset | n | Features | Non-default | n₀ | Source |
| --- | --- | --- | --- | --- | --- |
| [Default](https://doi.org/10.24432/C55S3H) | 30,000 | 23 | 77.9% | 5,000 | UCI ML Repository |
| [PPDai](https://figshare.com/articles/dataset/PPDaiData/13573856) | 31,389 | 29 | 77.0% | 5,000 | Figshare |
| [LendingClub](https://www.kaggle.com/datasets/marcusos/lending-club-clean) | 302,595 | 97 | 77.0% | 60,000 | Kaggle |

Raw data is not included in this repository. See [`data/README.md`](data/README.md) for acquisition instructions.

PPDai and LendingClub are down-sampled to a 77% non-default rate, producing an `_imb77` suffix in processed filenames; the `n` above is post-down-sampling. Default is used as distributed and is not down-sampled. LendingClub is additionally filtered to loans originated in 2017, with post-hoc leakage columns removed (Appendix A).

`n₀` is the initial training set size, read from `N0_VALUES` in `config.py`. These values were chosen by visual inspection of learning curves (Appendix C); that analysis is not included here.

## Repository Structure

```
├── data/
│   └── README.md                  # Dataset acquisition instructions
├── experiments/
│   ├── temporal_evolution/        # Section 4 results
│   └── exploration/               # Section 5 results
├── config.py                      # Paths, label columns, per-dataset settings, n0 values
├── data_loaders.py                # Loading raw and processed datasets
├── data_preprocess.py             # Per-dataset preprocessing (Appendix A)
├── download_data.py               # Automated download for Default; PPDai and LendingClub are manual
├── generate_downsampled.py        # Down-sampled dataset variants (imb77)
├── metrics.py                     # Accuracy, precision, recall, Kickout (Eq. 1-2)
├── models.py                      # GBDT and Random Forest construction, thresholding
├── reject_inference.py            # The five RI strategies and the two baselines (Section 3)
├── temporal_experiment.py         # Biased/Oracle lending simulation (Section 4.1)
├── temporal_experiment_ri.py      # Reject inference lending simulation (Section 4.3)
├── exploration_strategy.py        # Controlled exploration and the three samplers (Section 5)
├── run_preprocess.py              # Entry point: raw -> processed
├── run_temporal_experiments.py    # Entry point: Section 4
├── run_exploration_experiments.py # Entry point: Section 5
└── run_all_exploration.sh         # Driver for the full Section 5 sweep
```

## Reproducing the Experiments

The paper reports six configurations: three datasets × two models (`gbdt`, `rf`).

### Step 1: Acquire and preprocess the data

Follow [`data/README.md`](data/README.md), then:

```bash
python run_preprocess.py default ppdai lendingclub
python generate_downsampled.py
```

Run in this order. The experiments use the down-sampled variants, `ppdai_processed_imb77.csv` and `lendingclub_processed_imb77.csv`, which the second command produces.

Note that `run_preprocess.py` and `generate_downsampled.py` take dataset names as positional arguments, while the experiment runners below take `--datasets`.

### Step 2: Reject inference (Section 4)

Table 1, Figures 1–3, Appendix E.

```bash
python run_temporal_experiments.py \
    --datasets default --models gbdt rf --full-ri --n-runs 5

python run_temporal_experiments.py \
    --datasets ppdai lendingclub --models gbdt rf \
    --imbalance-ratio 0.77 --full-ri --n-runs 5
```

`--full-ri` enables the seven-strategy comparison. Without it, only the Biased and Oracle baselines are run (Figure 1). Writes to `experiments/temporal_evolution/results/`.

### Step 3: Controlled exploration (Section 5)

Tables 2–4, Figures 4–6, Appendix F.

```bash
bash run_all_exploration.sh
```

Runs 11 exploration rates × 3 sampling strategies × 5 runs × 10 cycles for all six configurations, handling the down-sampling split described above. For a smoke test:

```bash
bash run_all_exploration.sh --quick
```

Writes to `experiments/exploration/results/`.

## Results

### `experiments/temporal_evolution/results/`

`<dataset>_<model>_T10_n0<n0>_tau0.50_aggregated.csv` holds the Biased and Oracle baselines; `..._ri_aggregated.csv` holds all seven strategies. One row per lending cycle (`iteration`, 0–10), with columns `<strategy>_<metric>_mean` and `_std` over the five runs.

`<strategy>` is one of `filtered` (the Biased baseline), `oracle`, `parceling`, `fuzzy_augmentation`, `simple_extrapolation`, `shallow_self_learning`, `twins`. `<metric>` is one of `accuracy`, `precision`, `recall`, `kickout`, `default_rate`.

- `default_rate` is the training set default rate (TDR) of Section 3, computed over the training set accumulated up to that iteration.
- `accuracy`, `precision`, `recall` and `kickout` are computed on a held-out split of that training set, against the labels recorded in it. For the reject inference strategies those labels include the strategy's own imputations — see Section 3 and Limitations in the paper.
- `kickout` equals `2 · precision − 1` by construction (Eq. 2).

### `experiments/exploration/results/`

Three files per configuration:

- `exploration_<config>.csv` — one row per (ranking strategy, exploration rate, run, cycle)
- `summary_<config>.csv` — aggregated over runs; one row per (ranking strategy, exploration rate, cycle)
- `pareto_<config>.csv` — the final cycle (t = 10) only, one row per (ranking strategy, exploration rate). Despite the filename this is not a Pareto frontier; no dominance filtering is applied.

Three default rates are recorded per cycle. `training_default_rate` is the TDR over the accumulated training set — the same quantity as `default_rate` in the Section 4 files above. `reject_default_rate` is the true default rate among all applicants the model rejected at that cycle; it is a simulation diagnostic, not something a lender could observe, and is not reported in the paper. `exploration_default_rate` is the true default rate among the rejects approved for exploration (Table 4), and coincides with `reject_default_rate` at r = 1.0.

Tables 2, 3 and 4 are derived from `summary_<config>.csv` at t = 10.

## Contact

For questions or feedback, please contact Bruno Scarone at <scarone.b@northeastern.edu>.

## Citation

```bibtex
@inproceedings{scarone2026illusion,
  title     = {The Illusion of Improvement: Reject Inference Strategies in Credit Scoring},
  author    = {Scarone, Bruno and Baeza-Yates, Ricardo},
  booktitle = {Machine Learning and Knowledge Discovery in Databases},
  series    = {ECML PKDD '26},
  year      = {2026},
  publisher = {Springer},
  note      = {TODO: pages, DOI, volume}
}
```

## License

MIT License. See `LICENSE`.