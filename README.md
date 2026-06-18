# 🏦 Loan Default Prediction

A machine learning project that predicts whether a borrower is likely to default on a loan, built on a real-world dataset of 150,000 borrowers.

🚀 **Live Demo:** https://loan-default-prediction-7tb2mkxgzjjdkm7lkmmszx.streamlit.app/
---

## 📌 Problem Statement

Banks and financial institutions lose significant revenue when borrowers default on loans. Manually evaluating each loan application is slow and inconsistent. This project builds an automated ML pipeline that predicts loan default risk based on a borrower's financial profile — helping banks make faster, data-driven lending decisions.

---

## 🎯 Results

| Model | Accuracy | ROC-AUC | Default Recall |
|---|---|---|---|
| Logistic Regression | 66% | 0.70 | 0.65 |
| Random Forest | 89% | **0.82** | 0.45 |
| XGBoost | 88% | 0.82 | 0.49 |

✅ **Final Model: Random Forest** — selected for highest ROC-AUC and accuracy.

---

## 📂 Project Structure

```
loan-default-prediction/
│
├── 01_EDA.ipynb              ← Exploratory Data Analysis
├── 02_cleaning.ipynb         ← Data Cleaning & Preprocessing
├── 03_modelling.ipynb        ← Model Training & Evaluation
├── app.py                    ← Streamlit Web Application
├── requirements.txt          ← Dependencies
└── README.md
```

---

## 📊 Dataset

- **Source:** [Give Me Some Credit — Kaggle](https://www.kaggle.com/c/GiveMeSomeCredit)
- **Size:** 150,000 rows, 11 features
- **Target:** `SeriousDlqin2yrs` (1 = defaulted, 0 = did not default)

### Features Used

| Feature | Description |
|---|---|
| Age | Age of borrower |
| MonthlyIncome | Monthly income in USD |
| DebtRatio | Monthly debt payments / monthly income |
| RevolvingUtilizationOfUnsecuredLines | Credit card utilization ratio |
| NumberOfOpenCreditLinesAndLoans | Total open loans and credit lines |
| NumberRealEstateLoansOrLines | Number of mortgage/real estate loans |
| NumberOfDependents | Number of family dependents |
| NumberOfTime30-59DaysPastDueNotWorse | Times 30-59 days late on payment |
| NumberOfTime60-89DaysPastDueNotWorse | Times 60-89 days late on payment |
| NumberOfTimes90DaysLate | Times 90+ days late on payment |

---

## 🔍 Key Findings from EDA

- **Class Imbalance:** 93.3% non-defaulters vs 6.7% defaulters → handled using SMOTE
- **Missing Values:** MonthlyIncome had 19.82% missing → filled using median imputation
- **Age Outliers:** 14 records with age = 0 or age > 100 → removed
- **Income Outliers:** Maximum income was $3,008,750 → capped at $50,000
- **Most Common Age Group:** 40–60 years old

---

## ⚙️ ML Pipeline

```
Raw Data
   ↓
Exploratory Data Analysis
   ↓
Data Cleaning (missing values, outliers, bad rows)
   ↓
Train / Test Split (80/20, stratified)
   ↓
SMOTE on training data only (balanced to 50/50)
   ↓
Feature Scaling (StandardScaler for Logistic Regression)
   ↓
Model Training (Logistic Regression, Random Forest, XGBoost)
   ↓
Evaluation (ROC-AUC, Precision, Recall, F1)
   ↓
Streamlit Deployment
```

---

## 🚀 How to Run the App

### Step 1 — Clone the Repository
```bash
git clone https://github.com/penguin2029/loan-default-prediction.git
cd loan-default-prediction
```

### Step 2 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Download Dataset
- Download `cs-training.csv` from [Kaggle](https://www.kaggle.com/c/GiveMeSomeCredit)
- Rename it to `loan_data.csv`
- Place it in the project folder

### Step 4 — Run Notebooks in Order
```
01_EDA.ipynb
02_cleaning.ipynb
03_modelling.ipynb   ← this saves rf_model.pkl
```

### Step 5 — Launch the App
```bash
streamlit run app.py
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core programming language |
| Pandas & NumPy | Data manipulation |
| Matplotlib & Seaborn | Data visualization |
| Scikit-learn | ML models, preprocessing, evaluation |
| XGBoost | Gradient boosting model |
| Imbalanced-learn | SMOTE for class balancing |
| Streamlit | Web application deployment |
| Git & GitHub | Version control |

---

## 📈 Model Performance Deep Dive

### Why Random Forest Won
- Handles non-linear relationships between features
- Robust to outliers
- No feature scaling required
- Ensemble of 100 decision trees reduces overfitting

### Why Logistic Regression Underperformed
- Linear model — cannot capture complex patterns in financial data
- Achieved only 0.70 ROC-AUC despite SMOTE balancing

### Precision-Recall Tradeoff
- Random Forest: higher precision (0.29) → fewer false alarms
- XGBoost: higher recall (0.49) → catches more defaulters
- For real banking use, **recall is more important** — missing a defaulter costs more than a false alarm

---

## 👩‍💻 Author

**Shweta Prabhu**
- 📧 prabhushweta445@gmail.com
- 💼 [LinkedIn](https://linkedin.com/in/shwetaprabhu)
- 🐙 [GitHub](https://github.com/penguin2029)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
