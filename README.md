# 🩺 Diabetes Risk Classification — End-to-End Machine Learning Project

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-189914?style=for-the-badge&logo=xgboost&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)

[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Saikat1462/Diabetes-Risk-Prediction.git)
[![Live App](https://img.shields.io/badge/Live-Web_App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://diabetes-risk-prediction-8qsoycpbw3bvgj5b2ynqx6.streamlit.app/#diabetes-risk-predictor)

**A production-quality, end-to-end machine learning pipeline for binary classification of Diabetes risk using real-world CDC health survey data.**

**Author:** Saikat Sarkar &nbsp;|&nbsp; 🎓 IIT Jodhpur &nbsp;|&nbsp; 📊 Data Science & AI &nbsp;|&nbsp; [![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/saikat-sarkar-17151a3b1)

</div>

---

## 📌 Overview

This project builds a complete ML pipeline to predict whether a person has **Diabetes** or **No Diabetes** based on 21 behavioral and clinical health indicators collected from 253,680 US adults via the **CDC Behavioral Risk Factor Surveillance System (BRFSS)** survey.

The final model — a **Stacking Classifier** combining 5 diverse base estimators — is selected for its superior balance of Recall, AUC, and generalization, making it the most clinically appropriate choice for a healthcare screening context where **minimizing missed diagnoses (False Negatives) is the top priority**.

---

## 🎯 Problem Statement

> *Can we accurately identify individuals at risk of Diabetes using only self-reported health and lifestyle data — without requiring any lab tests?*

| Class | Label | Count |
|-------|-------|-------|
| `0` | No Diabetes | ~213,703 |
| `1` | Diabetes | ~35,346 |

> **Note:** Prediabetes (original class `1` in the raw dataset) was removed due to its extremely small sample size (~1.8%), reformulating this as a clean **binary classification** problem.

---

## 📁 Project Structure

```
Diabetes/
│
├── 📓 diabetes.ipynb          # Main notebook — full ML pipeline
├── 📊 diabetes.csv            # Raw dataset (CDC BRFSS)
├── 📦 requirements.txt        # All Python dependencies
│
├── 🤖 best_model_dt.pkl       # Saved Decision Tree model
├── 🤖 best_model_rf.pkl       # Saved Random Forest model
├── 🤖 best_model_knn.pkl      # Saved KNN model
├── 🤖 best_model_svm.pkl      # Saved Linear SVM model
├── 🤖 best_model_gb.pkl       # Saved Gradient Boosting model
├── 🤖 best_model_xgb.pkl      # Saved XGBoost model
└── 🤖 best_model_stack.pkl    # Saved Stacking Classifier (Final Model)
```

---

## 🔬 Dataset

| Attribute | Detail |
|-----------|--------|
| **Source** | [CDC BRFSS 2015 — Kaggle](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset) |
| **Rows** | 253,680 survey responses |
| **Features** | 21 health indicators |
| **Target** | Binary — Diabetes (1) / No Diabetes (0) |
| **Missing Values** | None |

### Key Features

| Feature | Type | Description |
|---------|------|-------------|
| `BMI` | Continuous | Body Mass Index |
| `HighBP` | Binary | High blood pressure flag |
| `HighChol` | Binary | High cholesterol flag |
| `GenHlth` | Ordinal | Self-rated general health (1=Excellent → 5=Poor) |
| `Age` | Ordinal | 13-level age bracket |
| `PhysHlth` | Continuous | Days of poor physical health (0–30) |
| `MentHlth` | Continuous | Days of poor mental health (0–30) |
| `Stroke` | Binary | Ever had a stroke |
| `HeartDiseaseorAttack` | Binary | Coronary heart disease or MI |

---

## 🛠️ Tech Stack

| Category | Libraries |
|----------|-----------|
| **Data Processing** | `pandas`, `numpy`, `scipy` |
| **Visualization** | `matplotlib`, `seaborn`, `plotly` |
| **Machine Learning** | `scikit-learn`, `xgboost`, `imbalanced-learn` |
| **Statistical Analysis** | `statsmodels` (VIF) |
| **Explainability** | `shap` |
| **Model Persistence** | `joblib` |

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Saikat1462/Diabetes-Risk-Prediction.git
cd Diabetes-Risk-Prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. (Optional) Install SHAP for Explainability
```bash
pip install shap
```

### 4. Launch Jupyter Notebook
```bash
jupyter notebook diabetes.ipynb
```

### 5. Load Pre-Trained Models (Skip Training)
All models are saved as `.pkl` files. To skip the grid search and load directly:
```python
import joblib
best_model_stack = joblib.load("best_model_stack.pkl")
predictions = best_model_stack.predict(x_test)
```

---

## 🌐 Streamlit Web App

An interactive web app is included for real-time diabetes risk prediction. 
**👉 [Try the Live App Here!](https://diabetes-risk-prediction-8qsoycpbw3bvgj5b2ynqx6.streamlit.app/#diabetes-risk-predictor)**

### Run Locally
```bash
streamlit run app.py
```

### Deploy to Streamlit Cloud
1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set **Main file path** to `app.py`
5. Click **Deploy** 🚀

### App Features
- 🎛️ **Interactive sidebar** — input all 21 health indicators with sliders and toggles
- 🎯 **Instant prediction** — risk probability + color-coded result card
- 📊 **Risk gauge** — visual probability meter
- 🚨 **Active risk factor list** — highlights which risk factors are present
- 📈 **Feature Importance chart** — XGBoost-based ranked feature plot
- ℹ️ **About tab** — model details, dataset info, limitations

---

## 📋 Notebook Pipeline

```
📓 diabetes.ipynb
│
├── Step 1   →  Load Dataset & Feature Dictionary
├── Step 2   →  Initial Data Inspection (dtypes, nulls, cardinality)
├── Step 3   →  Descriptive Label Encoding (for EDA only)
├── Step 4   →  Descriptive Statistics
├── Step 5   →  Target Class Distribution Analysis
├── Step 6   →  Continuous Variable Distributions (BMI, MentHlth, PhysHlth)
├── Step 7   →  Categorical & Binary Feature Distributions
├── Step 8   →  Bivariate Analysis: BMI vs. Diabetes Status
├── Step 9   →  Risk Factor Profiling by Diabetes Class
├── Step 10  →  Feature Correlation Heatmap
├── Step 11  →  Outlier Detection (IQR) + Medical Reasoning
├── Step 12  →  Multicollinearity Check (VIF)
├── Step 13  →  Model Preprocessing (Train/Test Split + SMOTE)
├── Step 14  →  Helper Functions (tune, evaluate)
│
├── Models Trained:
│   ├── 🌳  Decision Tree (GridSearchCV)
│   ├── 🌲  Random Forest (GridSearchCV)
│   ├── 👥  K-Nearest Neighbors (Pipeline + GridSearchCV)
│   ├── ⚡  Linear SVM (Pipeline + GridSearchCV)
│   ├── 🚀  Gradient Boosting (GridSearchCV)
│   ├── ⚡  XGBoost (GridSearchCV)
│   └── 🏆  Stacking Classifier (meta-learner = Logistic Regression)
│
├── Step 14  →  ROC Curve Comparison (all models)
├── Step 15  →  Precision-Recall Curve Comparison
├── Step 15  →  AUC Score Leaderboard
├── Step 16  →  Confusion Matrix (Stacking Classifier)
├── Step 17  →  Classification Report
├── Step 18  →  Feature Importance (XGBoost)
├── Step 19  →  Final Model Interpretation & Conclusions
├── Step 20  →  SHAP Global Summary + Patient-Level Waterfall Plot
└── Step 21  →  Limitations & Future Work
```

---

## 🏆 Model Performance Summary

> Sorted by **Recall (Class 1 — Diabetes)** — our primary clinical metric.

| Model | Recall (Diabetes) | Precision (Diabetes) | ROC-AUC | Macro F1 |
|-------|:-----------------:|:--------------------:|:-------:|:--------:|
| 🏆 **Stacking Classifier** | ✅ Best | ✅ Strong | ✅ Top | ✅ Balanced |
| ⚡ XGBoost | High | High | Top | High |
| 🚀 Gradient Boosting | High | Moderate | High | High |
| 🌲 Random Forest | Moderate | High | High | Moderate |
| 🌳 Decision Tree | Moderate | Moderate | Moderate | Moderate |
| ⚡ Linear SVM | Moderate | High | Moderate | Moderate |
| 👥 KNN | Lower | Moderate | Lower | Lower |

---

## 🏆 Why Stacking Classifier?

The Stacking Classifier was selected as the final model for 4 key reasons:

1. **Diversity** — Combines tree-based (DT, RF, XGB, GB) and linear (SVM) models, each capturing different patterns in the data
2. **Intelligent Combination** — A Logistic Regression meta-learner *learns* the optimal weights, rather than using a fixed rule
3. **No Data Leakage** — Built with 5-fold cross-validated out-of-fold predictions
4. **Clinical Priority** — Superior Recall on the Diabetes class minimizes missed diagnoses (False Negatives)

> In a medical screening context, a **False Negative** (missing a diabetic patient) carries far greater risk than a **False Positive** (flagging a healthy patient for follow-up tests).

---

## 🔑 Key Findings

- **`HighBP`** is the single strongest predictor of Diabetes risk — aligning with clinical evidence linking hypertension and Type 2 Diabetes
- **`BMI`** is a direct biological marker — diabetic patients have significantly higher median BMI
- **`GenHlth`** (self-rated health) is a powerful proxy — poor self-rated health strongly correlates with diabetes
- **`Age`** shows a clear monotonic trend — diabetes prevalence rises steeply with age
- **`HighChol`**, **`DiffWalk`**, and **`HeartDiseaseorAttack`** reflect the broader cardiovascular-metabolic risk cluster

---

## ⚠️ Limitations

| Limitation | Description |
|------------|-------------|
| **Self-Report Bias** | BRFSS responses may under/over-report sensitive behaviors |
| **SMOTE Caveats** | Synthetic samples may not reflect real patient profiles |
| **US-Only Data** | Model may not generalize to non-US populations |
| **LinearSVC** | Uses `decision_function()` as ROC/PR proxy (no `predict_proba`) |
| **Fixed Threshold** | All evaluations use 0.5 — threshold tuning may improve clinical recall |

---

## 🔭 Future Work

- [ ] **Threshold Optimization** — Tune decision boundary for maximum clinical recall
- [ ] **SHAP on Stacking Meta-Learner** — Explain how base model predictions are combined
- [ ] **External Validation** — Test on independent datasets (NHANES, UK Biobank)
- [ ] **Probability Calibration** — Apply `CalibratedClassifierCV` to LinearSVC
- [ ] **Feature Engineering** — Explore interaction terms (e.g., `BMI × HighBP`)
- [ ] **Longitudinal Analysis** — Track risk trajectory over multiple BRFSS survey years

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- **Dataset:** [CDC BRFSS Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset) via Kaggle
- **Survey Source:** [CDC Behavioral Risk Factor Surveillance System](https://www.cdc.gov/brfss/)

---

## 👨‍💻 Author

| | |
|--|--|
| **Name** | Saikat Sarkar |
| **Institution** | Indian Institute of Technology (IIT) Jodhpur |
| **Program** | Data Science & Artificial Intelligence |
| **LinkedIn** | [![LinkedIn](https://img.shields.io/badge/LinkedIn-Saikat%20Sarkar-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/saikat-sarkar-17151a3b1) |

---

<div align="center">

*Built with ❤️ by **[Saikat Sarkar](https://www.linkedin.com/in/saikat-sarkar-17151a3b1)** · IIT Jodhpur · Data Science & AI*

</div>
