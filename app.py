from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

# Predict page open cheyyadaniki
@app.route('/predict')
def predict_page():
    return render_template("predict.html")

# Form submit ayyaka result chupinchadaniki
@app.route('/result', methods=['POST'])
def result():
    prediction = "Loan Approved"
    return render_template("result.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)