# Notebook 04: Random Forest Baseline Model

## 📌 Summary
Notebook `04_random_forest_baseline.ipynb` constructs, trains, and evaluates an unweighted **Random Forest Classifier** to serve as the benchmark baseline for all subsequent model improvements.

---

## 🌲 Model Configuration
* **Classifier:** `sklearn.ensemble.RandomForestClassifier`
* **Parameters:** `random_state=42` (default parameters, unweighted)

```python
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
```

---

## 📈 Evaluation Results (Test Set: 2,000 samples)

### Confusion Matrix
```text
               Predicted Normal (0)   Predicted Failure (1)
Actual Normal (0)      1926                     6
Actual Failure (1)       26                    42
```

* **True Positives (TP):** 42
* **True Negatives (TN):** 1,926
* **False Positives (FP):** 6
* **False Negatives (FN):** 26 (Missed failures)

### Classification Metrics

| Class | Precision | Recall | F1-Score | Support |
| :--- | :-: | :-: | :-: | :-: |
| **No Failure (`0`)** | 0.99 | 1.00 | 0.99 | 1,932 |
| **Failure (`1`)** | **0.88** | **0.62** | **0.72** | **68** |
| **Accuracy** | — | — | **0.984 (98.4%)** | 2,000 |
| **Macro Average** | 0.93 | 0.81 | 0.86 | 2,000 |
| **Weighted Average** | 0.98 | 0.98 | 0.98 | 2,000 |

---

## 🔍 Key Performance Insights
1. **High Precision (88%):** When the baseline Random Forest predicts a failure, it is correct 88% of the time.
2. **Low Recall (62%):** The model misses **38% of actual machine failures** (26 out of 68), which poses a high operational risk in industrial settings.
3. **Accuracy Fallacy:** The high overall accuracy (98.4%) is deceptive due to the 96.6% majority class dominance.
