# Notebook 07: XGBoost Model Development & Benchmark Comparison

## 📌 Summary
Notebook `07_xgboost_comparison.ipynb` implements Extreme Gradient Boosting (XGBoost) classifiers, handles feature name sanitization for compatibility, applies imbalance scale weighting (`scale_pos_weight`), and benchmarks all models across the project lifecycle.

---

## 🛠️ Implementation Details

### Feature Sanitization
XGBoost raises a `ValueError` if feature names contain special characters like brackets (`[` or `]`). Feature column names were sanitized as follows:
```python
X.columns = [col.replace("[", "").replace("]", "").strip() for col in X.columns]
```

### Imbalance Ratio Calculation
The scale factor for positive weights is calculated as:
$$\text{scale\_pos\_weight} = \frac{N_{\text{negative}}}{N_{\text{positive}}} = \frac{7729}{271} \approx 28.52$$

```python
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
```

---

## 🏆 Full Benchmark Comparison Matrix

| Model | Accuracy | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) | ROC-AUC |
| :--- | :-: | :-: | :-: | :-: | :-: |
| **Random Forest Baseline** | 98.4% | 0.8750 | 0.6176 | 0.7241 | 0.9699 |
| **Random Forest Balanced** | 98.2% | 0.7656 | 0.7206 | 0.7424 | 0.9671 |
| **XGBoost Default** | **98.6%** | **0.8704** | **0.6912** | **0.7705** ⭐ | **0.9765** ⭐ |
| **XGBoost Weighted (`scale_pos_weight`)** | **98.1%** | **0.7465** | **0.7794** ⭐ | **0.7626** | **0.9675** |

---

## 🎯 Model Recommendations for Deployment

1. **Best Overall Model:** **XGBoost Default** achieves the highest F1-Score (**0.7705**) and highest ROC-AUC (**0.9765**).
2. **Best Proactive Maintenance Model:** **XGBoost Weighted** achieves the highest Recall (**77.94%**), detecting **53 out of 68 actual failures** in the test set.
3. **Deployment Strategy:** For critical equipment where missing a failure leads to expensive downtime, **XGBoost Weighted** is recommended to maximize failure detection.
