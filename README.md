# NYC Taxi Tip Prediction with Apache Spark

A distributed machine learning project comparing **Linear Regression** and **Gradient Boosted Trees** for predicting taxi tip amounts on the NYC Yellow Taxi dataset (March 2016, 12.2 million trips).

Built for the *Cloud Platforms and Architectures* course at the Department of Informatics, Ionian University (2026).

---

## Headline finding

> **Linear Regression outperforms Gradient Boosted Trees at every dataset size tested, with 12× lower training cost.**

The relationship between trip features and tip amount is dominated by a single near-linear signal (`tip ≈ ~18% of fare_amount`). Complex ensemble methods cannot exploit nonlinear structure that isn't there, while paying the full cost of trying. See [Results](#results) below for full details.

---

## Dataset

- **Source:** [NYC Yellow Taxi Trip Records](https://www.kaggle.com/datasets/elemento/nyc-yellow-taxi-trip-data?select=yellow_tripdata_2016-03.csv), March 2016
- **Publisher:** NYC Taxi and Limousine Commission (TLC)
- **Raw size:** 12,210,952 rows × 19 columns (~1.5 GB CSV)
- **After cleaning:** 7,937,904 rows (filtered to credit-card payments only - cash tips are not recorded by NYC TLC)
- **Target variable:** `tip_amount` (regression problem)

---

## Methods

| Method | Type | Library |
|---|---|---|
| Linear Regression (L2 regularized) | Linear, closed-form | Spark MLlib |
| Gradient Boosted Trees | Sequential ensemble of decision trees | Spark MLlib |

Both models run as Spark MLlib `Pipeline`s sharing identical preprocessing (one-hot encoding of categorical features, standard scaling of numeric features). The only difference is the final estimator stage.

**Experiments performed:**

1. Baseline comparison on full training set (6.35M rows)
2. Feature importance analysis (GBT)
3. Scaling experiments at 5%, 10%, 25%, 50%, 100% of the training data, holding the test set fixed

---

## Project structure

```
├── 📁 notebooks                # Run in order 01 → 05
│   ├── 📄 01_explore.ipynb         Load CSV, inspect, convert to Parquet
│   ├── 📄 02_clean.ipynb           Cleaning waterfall
│   ├── 📄 03_features.ipynb        Feature engineering
│   ├── 📄 04_models.ipynb          Linear Regression + GBT training
│   └── 📄 05_scaling.ipynb         Scaling experiments
├── 📁 results                  # Generated tables and plots
│   ├── 📄 cleaning_waterfall.csv
│   ├── 📄 feature_correlations.csv
│   ├── 📄 gbt_feature_importance.csv
│   ├── 📄 gbt_feature_importance_perslot.csv
│   ├── ⚙️ model_metrics.json
│   ├── 🖼️ scaling_plot.png
│   └── 📄 scaling_results.csv
├── 📁 src                      # Shared utilities imported by notebooks
│   ├── 🐍 __init__.py
│   └── 🐍 spark_utils.py
├── ⚙️ .gitignore
├── 📄 LICENSE
├── 📝 README.md
└── 📄 requirements.txt
```

---

## Setup

### Prerequisites

- Python 3.10 or newer
- Java 17 (Eclipse Temurin recommended; Java 21+ breaks PySpark 3.5)
- **Windows users only:** `winutils.exe` and `hadoop.dll` for Hadoop 3.3 from [cdarlint/winutils](https://github.com/cdarlint/winutils), placed in `C:\hadoop\bin\`, with environment variable `HADOOP_HOME=C:\hadoop`

### Installation

```bash
git clone https://github.com/georgopetrisg/taxi-analysis-project
cd taxi-analysis-project

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
.venv\Scripts\activate             # Windows PowerShell

# Install dependencies
pip install -r requirements.txt
```

### Data

The dataset is not committed (1.5 GB). Download `yellow_tripdata_2016-03.csv` from the [Kaggle source](https://www.kaggle.com/datasets/elemento/nyc-yellow-taxi-trip-data?select=yellow_tripdata_2016-03.csv) and place it at `data/yellow_tripdata_2016-03.csv`.

### Reproducing the results

Run the notebooks in order:

```bash
jupyter lab
```

Open `notebooks/01_explore.ipynb` and run all cells. Repeat for 02, 03, 04, 05. Each notebook reads outputs of the previous one.

**Approximate runtime on a 16 GB / 8-core laptop:** ~90 minutes total. Most of this is the GBT training in notebooks 04 and 05.

---

## Results

### Baseline comparison (full dataset, 6,350,117 training rows)

| Model | RMSE | MAE | R² | Train time |
|---|---:|---:|---:|---:|
| Linear Regression | **1.2971** | 0.7056 | **0.6857** | **16.2 s** |
| Gradient Boosted Trees | 1.3029 | **0.6977** | 0.6829 | 197.8 s |

LR wins on RMSE and R²; GBT wins on MAE. The MAE/RMSE gap shows GBT handles typical tips well but mispredicts large outliers more than LR does.

### Feature importance (top 5, Gradient Boosted Trees)

| Feature | Importance |
|---|---:|
| `fare_amount` | 85.8% |
| `tolls_amount` | 3.2% |
| `avg_speed_mph` | 2.8% |
| `pickup_hour` | 2.0% |
| `trip_distance` | 1.2% |

A single feature carries 86% of the model's predictive power. This explains the result above: when one feature dominates linearly, a linear model captures the structure as efficiently as a complex one.

### Scaling

See `results/scaling_plot.png`.

- **LR training time** is nearly flat below 1M rows (3-9 s), then scales roughly linearly with data.
- **GBT training time** scales from 31 s (5% data) to 198 s (100% data) - sub-linear growth dominated by fixed overhead at small sizes, approaching linear at larger sizes.
- **LR test RMSE** is essentially constant across the full data range (variance: 0.0008).
- **GBT test RMSE** improves monotonically with more data but does not surpass LR within the tested range.

---

## Limitations

- Single month of data (March 2016) - temporal generalization untested
- GBT hyperparameters not tuned via grid search
- Spark in local mode (single machine), not a true distributed cluster
- Cash trips excluded (NYC TLC does not record cash tips)

---

## Author

Georgios Georgopetris - inf2023031  
Ionian University - Department of Informatics

## License

This project was developed as university coursework.  

Distributed under the **MIT License**.  
See [LICENSE](LICENSE) for more information.