# Data

The three datasets are third-party and are not committed to this repository.
This file describes how to obtain and prepare them.

| Dataset | Key | Source | Raw file | Destination |
| --- | --- | --- | --- | --- |
| UCI Default of Credit Card Clients | `default` | [UCI](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) | `default_raw.xls` | `data/default/` |
| PPDai | `ppdai` | [Figshare](https://figshare.com/articles/dataset/DefaultData_and_PPDaiData/13573871) | `ppdaiData.csv` | `data/ppdai/` |
| LendingClub | `lendingclub` | [Kaggle](https://www.kaggle.com/datasets/marcusos/lending-club-clean) | `lending_club_clean.feather` | `data/lendingclub/` |

## 1. Download

**Default** downloads automatically:

```bash
python download_data.py default
```

It is an Excel file, so `xlrd` must be installed (it is in `requirements.txt`).

**PPDai** requires a manual download. Go to the Figshare link above, download
`ppdaiData.csv`, and place it in `data/ppdai/`.

**LendingClub** requires a Kaggle account. Either use the CLI:

```bash
pip install kaggle
# Place your API key at ~/.kaggle/kaggle.json

kaggle datasets download -d marcusos/lending-club-clean
mv lending_club_clean.feather data/lendingclub/
```

or download `lending_club_clean.feather` from the Kaggle link above and place it
in `data/lendingclub/`.

## 2. Preprocess

```bash
python run_preprocess.py default ppdai lendingclub
```

This applies the per-dataset preprocessing of Appendix A — label construction,
the LendingClub 2017 filter and leakage-column removal, null handling — and
writes `<key>_processed.csv` into each dataset directory.

Called with no arguments, `run_preprocess.py` processes every dataset whose raw
file is present. `python run_preprocess.py --list` shows the available keys.

## 3. Generate the down-sampled variants

```bash
python generate_downsampled.py
```

The experiments use PPDai and LendingClub down-sampled to a 77% non-default
rate, which this produces as `ppdai_processed_imb77.csv` and
`lendingclub_processed_imb77.csv`. Default is used as distributed (77.9%
non-default) and is not down-sampled.

Run this after step 2.

## Expected directory structure

```
data/
├── README.md
├── default/
│   ├── default_raw.xls                  ← download
│   └── default_processed.csv            ← run_preprocess.py
├── ppdai/
│   ├── ppdaiData.csv                    ← download
│   ├── ppdai_processed.csv              ← run_preprocess.py
│   └── ppdai_processed_imb77.csv        ← generate_downsampled.py
└── lendingclub/
    ├── lending_club_clean.feather       ← download
    ├── lendingclub_processed.csv        ← run_preprocess.py
    └── lendingclub_processed_imb77.csv  ← generate_downsampled.py
```

The experiments read `default_processed.csv`, `ppdai_processed_imb77.csv` and
`lendingclub_processed_imb77.csv`.