# Customer Churn Prediction — ANN (Kaggle Churn Modelling)

This repository contains a concise end‑to‑end workflow (in a single Jupyter notebook) to predict **bank customer churn** using a **feed‑forward Artificial Neural Network (ANN)** built with Keras/TensorFlow.

> **Notebook:** `Churn-customer.ipynb`

---

## 🧠 What this project does

1. **Fetches the dataset automatically** using `kagglehub`:
   - Dataset slug: `churn-modelling`
   - File used: `Churn_Modelling.csv`

2. **Preprocesses data**
   - Drops categorical fields to keep only numeric features:
     - Dropped columns: `RowNumber`, `CustomerId`, `Surname`, `Geography`, `Gender`
   - Separates **features** `x` (all columns except the target) and **target** `y` (`Exited`).
   - Standardizes features with **`StandardScaler`** (fit on the full `x` then transformed; the scaled values replace `x`).

3. **Train / Test Split**
   - Uses `train_test_split` with `test_size=0.2` and `random_state=42` to ensure reproducibility.

4. **Model: Keras Sequential ANN**
   - Architecture (input dimension = 8 features after dropping):
     - Dense(6, activation=`relu`)
     - Dense(4, activation=`relu`)
     - Dense(2, activation=`relu`)
     - Dense(1, activation=`sigmoid`)
   - Compilation: `optimizer='adam'`, `loss='binary_crossentropy'`, `metrics=['accuracy']`
   - Training: `batch_size=100`, `epochs=50`

5. **Evaluation**
   - Generates predictions on **test** and **train** sets.
   - Applies a **0.5 probability threshold** to convert predicted probabilities to class labels.
   - Computes **Accuracy** using `sklearn.metrics.accuracy_score` for both splits.

6. **Single‑row Inference (example)**
   - Creates a numpy array `input_data` with **1 row × 8 columns** (already **scaled** values) and calls `ann.predict(input_data)`.

> ⚠️ **Note on features**: In this notebook, `Geography` and `Gender` were **dropped** instead of being encoded. This keeps the pipeline purely numeric but may slightly limit performance versus one‑hot encoding those columns.

---

## 📁 Project structure

```
.
├── Churn-customer.ipynb   # the complete workflow (data download → preprocessing → ANN → evaluation → inference)
└── README.md              # this file
```

---

## 🚀 Quickstart

### 1) Environment
Create a virtual environment (recommended) and install dependencies:

```pip install   numpy   pandas   scikit-learn   tensorflow   kagglehub```

> If `tensorflow` (GPU) is desired, follow the official TensorFlow installation guide for your platform.

### 2) Run the notebook

Open Jupyter and run all cells in order:

```bash
jupyter notebook
# then open "Churn-customer.ipynb" and Run All
```

On first run, `kagglehub` will automatically download the dataset:
```python
import kagglehub
path = kagglehub.dataset_download("churn-modelling")
```
The file `Churn_Modelling.csv` is read from the downloaded path.

> **Offline alternative:** If you cannot download through `kagglehub`, manually place `Churn_Modelling.csv` in a local folder and change the `read_csv` path accordingly.

---

## 🧪 Reproducing results

- The data split is deterministic, with `random_state=42`, ensuring consistent results.
- Accuracy is computed on both **train** and **test** sets after converting probabilities to labels with a **0.5** threshold.

###  Final Test Scores:

- Training Accuracy: 84.15%

- Test Accuracy: 84.05%

> Exact scores depend on your environment and TensorFlow/Keras version. Check the last cells of the notebook for the printed values on your machine.



## 🔎 Key implementation details (mirrors the notebook)

- **Columns removed:** `RowNumber`, `CustomerId`, `Surname`, `Geography`, `Gender`
- **Target:** `Exited`
- **Scaling:** `StandardScaler` applied to `x`, then reassigned back to `x` (`pd.DataFrame(..., columns=x.columns)`)
- **Split:** `train_test_split(x, y, test_size=0.2, random_state=42)`
- **Model definition:**
  ```python
  from keras.models import Sequential
  from keras.layers import Dense

  ann = Sequential()
  ann.add(Dense(6,  input_dim=8, activation='relu'))
  ann.add(Dense(4,  activation='relu'))
  ann.add(Dense(2,  activation='relu'))
  ann.add(Dense(1,  activation='sigmoid'))

  ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
  ann.fit(x_train, y_train, batch_size=100, epochs=50)
  ```
- **Inference on a single row** (already-scaled input):
  ```python
  import numpy as np
  input_data = np.array([[-0.564197, -0.660018, -0.695982, 0.324119, 0.807737, -1.547768, -1.030670, -1.0138]])
  ann.predict(input_data)
  ```

---

## ✅ Requirements
- Packages: `numpy`, `pandas`, `scikit-learn`, `tensorflow`, `kagglehub`

You can export your environment with:
```bash
pip freeze > requirements.txt
```

---

## 📚 Dataset

- Source: Kaggle — **Churn Modelling** (`/churn-modelling`)
- The dataset is owned by the original author/uploader; please review and respect the dataset’s license on Kaggle.

---

## 📜 License & Use

- This code is provided for educational and research purposes.
- The dataset remains subject to its original Kaggle license/terms.
- No warranty; use at your own risk.

