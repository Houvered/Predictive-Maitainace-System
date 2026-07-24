# Notebook 03: Model Preparation & Train-Test Splitting

## 📌 Summary
Notebook `03_model_preparation.ipynb` sets up the training feature matrix $X$ and target vector $y$, prevents data leakage by excluding breakdown sub-type flags, and splits the data using stratified sampling into training and testing sets.

---

## 🎯 Feature & Target Definition

### Target Vector ($y$)
* `Machine failure` (`0` = No Failure, `1` = Failure)

### Input Feature Matrix ($X$)
The model is trained strictly on operational sensor telemetry and machine type:
1. `Type` (Encoded: 0, 1, 2)
2. `Air temperature [K]`
3. `Process temperature [K]`
4. `Rotational speed [rpm]`
5. `Torque [Nm]`
6. `Tool wear [min]`

### 🛑 Prevention of Target Leakage
The specific failure mode flags (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) are **dropped** from $X$. In a real-world predictive maintenance scenario, these flags are generated simultaneously when a failure occurs. Including them in $X$ would cause target leakage and unrealistically inflate model accuracy.

```python
X = df.drop(["Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"], axis=1)
y = df["Machine failure"]
```

---

## ✂️ Train-Test Split Configuration
* **Split Ratio:** 80% Training / 20% Testing
* **Stratified Sampling:** `stratify=y` is enforced so that both train and test sets maintain the exact ~3.39% minority class failure ratio.
* **Random State:** `42` (Ensures reproducibility)

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

### Dataset Dimensions & Class Distribution

| Split | Total Samples | Normal (`0`) | Failure (`1`) | Failure Ratio |
| :--- | :-: | :-: | :-: | :-: |
| **Training Set (`X_train`)** | 8,000 | 7,729 | 271 | 3.39% |
| **Testing Set (`X_test`)** | 2,000 | 1,932 | 68 | 3.40% |
