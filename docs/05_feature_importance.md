# Notebook 05: Feature Importance Analysis

## 📌 Summary
Notebook `05_feature_importance.ipynb` extracts and visualizes the Gini impurity importance scores from the trained Random Forest baseline model to identify which machine operating parameters have the strongest influence on predicting machine failures.

---

## 📊 Feature Importance Breakdown

| Rank | Feature Name | Importance Score | Percentage |
| :-: | :--- | :-: | :-: |
| 1 | **Torque [Nm]** | `0.3301` | **33.0%** |
| 2 | **Rotational speed [rpm]** | `0.2238` | **22.4%** |
| 3 | **Tool wear [min]** | `0.1644` | **16.4%** |
| 4 | **Air temperature [K]** | `0.1304` | **13.0%** |
| 5 | **Process temperature [K]** | `0.1272` | **12.7%** |
| 6 | **Type** | `0.0240` | **2.4%** |

---

## 🛠️ Mechanical & Physical Interpretation

1. **Torque (33.0%) & Speed (22.4%):**
   * Mechanical power is defined as $\text{Power} = \text{Torque} \times \text{Rotational Speed}$. Sudden spikes or unnatural inverse relationships between torque and speed are primary indicators of mechanical strain, stall conditions, and power overload failures.
2. **Tool Wear (16.4%):**
   * Cumulative wear time directly degrades friction and cutting efficiency. High tool wear increases strain on the drive motor, leading to overstrain or tool breakage failures.
3. **Air & Process Temperature (~25.7% Combined):**
   * The temperature difference $\Delta T = T_{\text{process}} - T_{\text{air}}$ indicates thermal dissipation efficiency. Excessive heat build-up contributes to heat dissipation failures (HDF).
4. **Machine Type (2.4%):**
   * Product quality class (`L`, `M`, `H`) has minimal direct impact on machine failure compared to physical operational telemetry.
