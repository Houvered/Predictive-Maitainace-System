# Notebook 06: Balanced Random Forest & Class Weighting

## 📌 Summary
Notebook `06_balanced_random_forest.ipynb` investigates methods to address class imbalance by configuring `class_weight='balanced'` within the Random Forest Classifier.

---

## ⚙️ Model Setup
```python
rf_balanced = RandomForestClassifier(
    class_weight="balanced",
    random_state=42
)
rf_balanced.fit(X_train, y_train)
```

The `balanced` mode automatically adjusts weights inversely proportional to class frequencies in the input data:
$$\text{Weight}_c = \frac{N_{\text{samples}}}{N_{\text{classes}} \times N_c}$$

---

## 📈 Performance Results & Comparison

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
| :--- | :-: | :-: | :-: | :-: |
| **No Failure (`0`)** | 0.99 | 0.99 | 0.99 | 1,932 |
| **Failure (`1`)** | **0.77** | **0.72** | **0.74** | **68** |

### Baseline vs. Balanced Random Forest

| Model Variant | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) |
| :--- | :-: | :-: | :-: |
| **Random Forest Baseline** | **0.88** | **0.62** | **0.72** |
| **Random Forest Balanced** | **0.77** | **0.72** ⭐ | **0.74** ⭐ |

---

## 💡 Key Findings
* **Increased Recall:** Setting `class_weight='balanced'` successfully increased failure recall from **62% (42 detected)** to **72% (49 detected)**.
* **Precision Trade-off:** Precision decreased from **88%** to **77%**, resulting in a small rise in False Positives.
* **Overall Benefit:** F1-Score improved from **0.72** to **0.74**, demonstrating that penalizing minority class misclassifications helps the model detect more actual failures.
