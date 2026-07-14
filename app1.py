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

# Load the trained model and scaler
model_path = os.path.join(BASE_DIR, "Training", "rdf.pkl")
scaler_path = os.path.join(BASE_DIR, "Training", "scale.pkl")

model = pickle.load(open(model_path, "rb"))
scaler = pickle.load(open(scaler_path, "rb"))


@app.route('/')
def home():
    return render_template("home.html")


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    return render_template("predict.html")


@app.route('/submit', methods=['POST'])
def submit():

    try:
        # Read values from HTML form in the expected feature order
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

        input_features = []
        for name in names:
            value = request.form.get(name)
            try:
                input_features.append(int(value))
            except (ValueError, TypeError):
                input_features.append(float(value))

        # Convert to DataFrame and apply the same scaling used during training
        data = pd.DataFrame([input_features], columns=names)
        data = scaler.transform(data)

        # Predict
        prediction = model.predict(data)
        result = prediction[0]

        if result in [1, 'Y', 'y', 'Yes', 'yes', 'Approved', 'approved']:
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