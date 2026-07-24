# Notebook 01: Data Understanding & Exploratory Data Analysis

## 📌 Summary
Notebook `01_data_understanding.ipynb` performs initial Exploratory Data Analysis (EDA) on the AI4I 2020 Predictive Maintenance Dataset. It verifies data dimensions, checks for missing values or duplicates, and analyzes the distribution of machine failure labels and operational sensor telemetry.

---

## 📊 Dataset Metadata & Structure
* **Raw File Path:** `data/raw/ai4i2020.csv`
* **Total Records:** 10,000 samples
* **Total Columns:** 14 columns

### Column Schema & Data Types

| # | Column Name | Data Type | Description |
| :-: | :--- | :-: | :--- |
| 0 | `UDI` | `int64` | Unique Identifier (1 to 10,000) |
| 1 | `Product ID` | `str` | Product variant serial number (e.g., `M14860`, `L47181`) |
| 2 | `Type` | `str` | Product quality class (`L` = Low [60%], `M` = Medium [30%], `H` = High [10%]) |
| 3 | `Air temperature [K]` | `float64` | Ambient air temperature in Kelvin (Min: 295.3 K, Max: 304.5 K) |
| 4 | `Process temperature [K]` | `float64` | Process temperature in Kelvin (Min: 305.7 K, Max: 313.8 K) |
| 5 | `Rotational speed [rpm]` | `int64` | Spindle rotational speed (Min: 1,168 rpm, Max: 2,886 rpm) |
| 6 | `Torque [Nm]` | `float64` | Motor torque in Newton-meters (Min: 3.8 Nm, Max: 76.6 Nm) |
| 7 | `Tool wear [min]` | `int64` | Accumulated tool wear time in minutes (Min: 0 min, Max: 253 min) |
| 8 | `Machine failure` | `int64` | **Target Variable** (`0` = No Failure, `1` = Failure) |
| 9 | `TWF` | `int64` | Tool Wear Failure mode binary flag |
| 10 | `HDF` | `int64` | Heat Dissipation Failure mode binary flag |
| 11 | `PWF` | `int64` | Power Failure mode binary flag |
| 12 | `OSF` | `int64` | Overstrain Failure mode binary flag |
| 13 | `RNF` | `int64` | Random Failure mode binary flag |

---

## 🔍 Data Quality Check Results
* **Missing Values:** `0` (100% complete dataset)
* **Duplicate Rows:** `0`
* **Target Distribution:**
  * `0` (No Failure): **9,661 samples (96.61%)**
  * `1` (Machine Failure): **339 samples (3.39%)**
* **Class Imbalance Ratio:** ~28.5 to 1 (Severe class imbalance)

---

## 🔑 Key Engineering Observations
1. The target variable is severely imbalanced, meaning standard accuracy is an inadequate evaluation metric.
2. Machine quality types are divided into Low (6000), Medium (2997), and High (1003).
3. The dataset includes 5 individual failure modes (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`), which must be excluded during model training to avoid target leakage.
