import os
import numpy as np
import pickle
import pandas as pd
from flask import Flask, request, render_template

# Initialize Flask application
app = Flask(
    __name__,
    template_folder=os.path.join("Flask", "templates"),
    static_folder=os.path.join("Flask", "static")
)

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the trained model
model_path = os.path.join(BASE_DIR, "Training", "rdf.pkl")
model = pickle.load(open(model_path, "rb"))


@app.route('/')
def home():
    return render_template("home.html")


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    return render_template("predict.html")


@app.route('/submit', methods=['POST'])
def submit():

    try:
        # Read values from HTML form
        input_features = []

        for value in request.form.values():
            try:
                input_features.append(int(value))
            except ValueError:
                input_features.append(float(value))

        names = [
            'Gender',
            'Married',
            'Dependents',
            'Education',
            'Self_Employed',
            'ApplicantIncome',
            'CoapplicantIncome',
            'LoanAmount',
            'Loan_Amount_Term',
            'Credit_History',
            'Property_Area'
        ]

        # Convert to DataFrame
        data = pd.DataFrame([input_features], columns=names)

        # Predict
        prediction = model.predict(data)

        if prediction[0] == 1:
            prediction_text = "Congratulations! Your Loan is Approved."
        else:
            prediction_text = "Sorry! Your Loan Application is Rejected."

        return render_template(
            "submit.html",
            prediction_text=prediction_text
        )

    except Exception as e:
        return f"Error: {e}"


if __name__ == "__main__":
    app.run(debug=True)