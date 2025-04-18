# app.py

from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load('model/heart_model.pkl')
scaler = joblib.load('model/scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict')
def form():
    return render_template('form.html')

@app.route('/result', methods=['POST'])
def result():
    try:
        features = [float(x) for x in request.form.values()]
        data = np.array(features).reshape(1, -1)
        data_scaled = scaler.transform(data)
        prediction = model.predict(data_scaled)
        risk = "High Risk of Heart Disease" if prediction[0] == 1 else "Low Risk of Heart Disease"
        return render_template('result.html', prediction=risk)
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
