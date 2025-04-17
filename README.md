
# 💳 Credit Card Fraud Detection using XGBoost & Random Forest

An end-to-end notebook for **fraud detection** using the **Kaggle Credit Card Dataset**. This project tackles **class imbalance** using **SMOTE**, evaluates models using **ROC AUC**, and performs **threshold optimization** for real-world fraud prediction.

---

## ✅ Features
- Handles highly **imbalanced data**
- Uses **SMOTE** for oversampling
- Implements **Random Forest** & **XGBoost**
- Includes **threshold tuning** for high recall
- **Real-time fraud prediction** function

---

## 📦 Dependencies

```bash
pip install numpy pandas scikit-learn imbalanced-learn matplotlib seaborn xgboost
```

---

## 📊 Dataset

- Publicly available from [TensorFlow](https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv)
- Contains anonymized features (`V1` to `V28`), `Amount`, `Time`, and `Class` (0: Normal, 1: Fraud)

---

## 📈 Steps Covered

### 1️⃣ Setup & Data Loading
- Install packages
- Load CSV from URL
- Display class distribution

### 2️⃣ Exploratory Analysis
- Visualize fraud vs non-fraud counts
- Understand class imbalance

### 3️⃣ Feature Engineering
- Normalize `Amount`
- Drop `Time`

### 4️⃣ Handling Imbalance
- Split train/test using `stratify=y`
- Apply **SMOTE** to training data only

### 5️⃣ Model Training
- **Random Forest** (with `class_weight='balanced'`)
- **XGBoost** (with `scale_pos_weight`)

### 6️⃣ Model Evaluation
- Classification report
- ROC AUC score
- Confusion matrix visualization

### 7️⃣ Threshold Optimization
- Uses `precision_recall_curve`
- Find threshold for **90% recall**

### 8️⃣ Real-Time Prediction
```python
def predict_fraud(transaction_data, model=xgb, threshold=0.5):
    """Predict fraud for new transaction"""
    proba = model.predict_proba(transaction_data)[0,1]
    return proba > threshold, proba
```

---

## 🎯 Why This Works
- Applies best practices to avoid data leakage
- Balances interpretability and performance
- Realistic example for fraud detection pipelines

---

## 📁 Example Output

```python
Prediction: Fraud
Probability: 0.9182
```

---

**Just run the notebook and plug in your data!**
```
