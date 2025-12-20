# Loan Approval Prediction System

---
This project predicts whether a loan application will be approved or rejected using a Machine Learning model. The system is built usign Logistic Regression and deployed as a web application using Flask.
---

## Home Page

![Home Page](images/Home%20Page.png)

## Input Form

![Form page](images/Form%20Page.png)

## Prediction Result
g
![Prediction Result](images/Result%20Page.png)


---
## Teckh Stack

- Python 3.11.3
- Flask (Web framework)
- Scikit-Learn (ML model)
- HTML, CSS & JS 


---

## How to Run Locally

1. **Clone the repo**
```bash
git clone https://github.com/sourov-roy-ovi/Loan-Prediction-Model.git
```

## Navigate into folder
cd Loan-Prediction-Model

## Create & activete virtual environment
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

# Notes
- This app predicts loan approval based on user inputs.
- Useful project for ML portfolio and beginner Python/Flask.
- Update model and UI for improvement.

--