# Notebook 02: Data Preprocessing Pipeline

## 📌 Summary
Notebook `02_preprocessing.ipynb` transforms the raw dataset into a machine-learning-ready clean format. It drops uninformative identifier columns, encodes categorical machine types into numerical values, and exports the clean dataset to disk.

---

## ⚙️ Transformations Performed

### 1. Identifier Removal
The columns `UDI` (index number) and `Product ID` (string serial code) are dropped because they represent arbitrary record identifiers and contain no predictive signal.
```python
df = df.drop(["UDI", "Product ID"], axis=1)
```

### 2. Categorical Encoding (`Type`)
The machine quality column `Type` contains text categories (`L`, `M`, `H`). Machine learning algorithms require numerical inputs. An ordinal mapping is applied based on product tier:
* `L` (Low Quality) $\rightarrow$ `0` (6,000 samples)
* `M` (Medium Quality) $\rightarrow$ `1` (2,997 samples)
* `H` (High Quality) $\rightarrow$ `2` (1,003 samples)

```python
type_mapping = {"L": 0, "M": 1, "H": 2}
df["Type"] = df["Type"].map(type_mapping)
```

---

## 💾 Output Artifact
The clean, encoded dataset is saved as a CSV file for downstream model training:
* **Output Path:** `data/processed/processed_ai4i.csv`
* **Dimensions:** 10,000 rows $\times$ 12 columns
* **Columns Remaining:** `Type`, `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`, `Machine failure`, `TWF`, `HDF`, `PWF`, `OSF`, `RNF`.
