# 🏦 Loan Approval Prediction System (Production-Ready ML Baseline)

---
## 📌 Project Overview
The **Loan Approval Prediction System** is an end-to-end Machine Learning web application that predicts whether a loan application will be **approved or rejected** based on applicant information.

This project follows an **industry-standard ML workflow**, including:
    - Data analysis & cleaning
    - Feature engineering
    - Model training & evaluation
    - Bias handling & threshold tuning
    - Model deployment using Flask
The final model is a **Balanced Logistic Regression** model deployed as a **production-ready-baseline(v1).**

---

## Business Problem
Banks and financial institutions need an automated system to:
    - Reduce manual loan screening effort
    - Minimize biased approvals
    - Improve decision consistency
This system predicts loan approval using historical applicant data.

---

## Machine Learning Approach
- **Algorithm** Logistic Regression (Balanced)

- **Target Variable**
    - 1 -> Approved
    - 0 -> Rejected

---

# Dataset & Exploratory Data Analysis (EDA)

## Data Inspection
- Checked dataset structure and statistics:
    - `df.info()`
    - `df.describe()`
    - Missing values using `df.isna().sum()`
- Duplicate check:
    - `df.duplicated().sum()` -> **0 duplicates found**

## Data Visualization
- Target variable distribution (Loan_Status)
- Numerical feature distributions (Histogram)
- Categorical vs Target analysis using **Countplot**

---

# Data Cleaning & Preprocessing

## Missing Value Handling
    - Missing values handled using appropriate statiscal methods

## Outlier Detection ( Without Deleting Data)
- Visual inspection using **Boxplots**
- Outliers identified using **IQR method**
- **Log Transformation** applied to numerical features
- Post-transformation distribution validated using **Histograms**

---

# Feature  Engineering

## Dropped Irrelevant Feature
- Loan_ID (no predictive value)

## Created New meaningful Features
- Total_Income
- EMI
- Income_Loan_Ratio
These features improved model interpretability and performance.

---

## Categorical Encoding
- Categorical features encoded using appropriate encoding techniques
- Ensured compatibility with Logistic Regression

---

## Feature Selection
Final selected features used for model training:
```python
final_features = [
    'Credit_History',
    'Property_Area_Semiurban',
    'Married',
    'Education',
    'CoapplicantIncome_log',
    'ApplicantIncome_log',
    'Income_loan_Ratio',
    'Dependents',
    'Gender'
]
```

---

## Model Training Pipeline
### Data Preparation
- Feature -> x
- Target -> y
- Train-test-split
- Feature scaling applied

### Initial Model
- Logistic Regression
- Evaluated using:
    - Accuracy Score
    - Confusion Matrix
    - Classification Report

### Bias Detection
- Observed class imbalance bias

---

## Bias Handling & Model Improvement
To reduce bias:
- Used **class_weight='balanced'**
- Tuned decision threshold
```python
model_balanced = LogisticRegression(
    max_iter=1000,
    class_weight='balanced'
)

model_balanced.fit(x_train_scaled, y_train)

y_prob = model_balanced.predict_proba(x_test_scaled)[:, 1]
y_pred_60 = (y_prob > 0.6).astype(int)
```

## Result
- Slight accuracy reduction
- **Significant improvement in class balance**
- More reliable real-world predictions

---

## Model Evaluation
- Accuracy Score
- Confusion Matrix (visualized)
- Classification Report

### 📌 Conclusion:
Accuracy alone is not sufficient. Balanced predictions were prioritized for production use.

---

## Model Saving
- Final trained model saved using `pickle`
- Loaded during Flask app runtime

---

## Web Application Deployment (Flask)
### Backend
- Flask framework
- Model inference API

### Frontend
* `index.html` -> Home Page
* `predict.html` -> Input Form
* `result.html` -> Prediction Result
* Styled using `CSS`
* Interactive behavior using `JavaScript`

---

# Application Screenshots

## Home Page
<imag src= "images/Home Page.png" alt="Home Page Screenshot" width="600">

## Input Form

<img src="images/Form Page.png" alt="Form Page Screenshot" width="600">

## Prediction Result
g
<img src="images/Result Page.png" alt="Result Page Screenshot" width="600">


---
## Tech Stack

- Python 3.11.3
- Flask
- Scikit-Learn 
- Pandas, Numpy
- Matplotlib, Seaborn
- HTML, CSS & JavaScript
- Git & GitHub

---

## Why Logistic Regression?

- Simple and interpretable
- Works well with structured financial data
- Suitable as a strong baseline model
- Easier to monitor and debug in production

---

## How to Run Locally

1. **Clone the repo**
```bash
git clone https://github.com/sourov-roy-ovi/Loan-Prediction-Model.git
```

## Navigate into folder
cd Loan-Prediction-Model

## Create & activate virtual environment
python -m venv venv
venv\Scripts\activate

## Install dependencies
pip install -r requirements.txt

## Run the app
python app.py

---

# Project Structure
```Loan-Prediction-ML/
│
├── data/
│   └── loan_data_set.csv
│
├── notebooks/
│   └── EDA_and_Model_Training.ipynb
│
├── model/
│   └── loan_model.pkl
│
├── app.py
│
├── templates/
│   ├── index.html
│   ├── predict.html
│   └── result.html
│
├── static/
│   ├── style.css
│   └── script.js
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Versioning Strategy
* v1 (Current)
    - Balanced Logistic Regression
    - Threshold = 0.6
    - Flask deployment
    - Industry-grade ML pipeline
**Production-ready baseline**

---

## Future Improvements
- Try advanced models (Random Forest, XGBoost)
- Hyperparameter tuning
- Model explainability (SHAP)

---

# 👨‍💻 Author
**Sourov Roy**
Aspiring Machine Learning Engineer
Focused on building real-world ML systems

---
