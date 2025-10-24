# MVP: Penguin Similarity (Cosine on Standardized Features)

## Quick Start (works if `penguins_lter.csv` is in this folder)

```bash
# from the project root (this folder):
python -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/run_similarity_penguins.py
```

Outputs will appear in `outputs/`:
- `top10_all_queries.csv` (Top-10 for each of 3 queries)
- `pca_scatter.png`
- `feature_bars.png`
- `validation_summary.csv`

## Project Layout
```
.
├── penguins_lter.csv              # put the dataset here (this file is required)
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── penguins_clean.csv         # created by the script
├── outputs/                       # created by the script
│   ├── top10_all_queries.csv
│   ├── pca_scatter.png
│   ├── feature_bars.png
│   └── validation_summary.csv
└── scripts/
    └── run_similarity_penguins.py
```

## Notes
- The script accepts the LTER CSV with column names like `Species`, `Culmen Length (mm)`, etc., or a tidy CSV with lowercase names; it handles both.
- If you move the CSV, update `DATA_PATH` inside `scripts/run_similarity_penguins.py` to point to the new location.
- No other configuration needed.
