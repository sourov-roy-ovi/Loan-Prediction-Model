from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__)

Model_dir = "model"
model = joblib.load(os.path.join(Model_dir, "loan_model_v1.pkl"))
scaler = joblib.load(os.path.join(Model_dir, "scaler.pkl"))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict_page")
def predict_page():
    return render_template("predict.html")

@app.route("/predict", methods=["POST"])
def predict():

    credit_history = float(request.form['Credit_History'])
    property_area = float(request.form['Property_Area_Semiurban'])
    married = float(request.form['Married'])
    education = float(request.form['Education'])
    dependents = float(request.form['Gender'])
    gender = float(request.form['Gender'])


    coapp_income_log = np.log(float(request.form['CoapplicantIncome']) + 1)
    app_income_log = np.log(float(request.form['Income_loan_Ratio']))

    income_loan_ratio = float(request.form['Income_loan_Ratio'])

    input_data = np.array([[
        credit_history,
        property_area,
        married,
        education,
        coapp_income_log,
        app_income_log,
        income_loan_ratio,
        dependents,
        gender
    ]])


    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    result = "Your Loan Approved" if prediction == 1 else "Your Loan Rejected"
    return render_template("result.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)