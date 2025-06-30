from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model
model = joblib.load('CARDIO.joblib')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/info')
def info():
    return render_template('info.html')


@app.route('/main')
def main():
    return render_template('main.html', errors=[], form_data={})


@app.route('/predict', methods=['POST'])
def predict():
    errors = []
    form_data = {}
    try:
        age = int(request.form['age'])
        if not (25 <= age <= 80):
            errors.append("Age must be between 25 and 80.")
        form_data['age'] = age

        gender = int(request.form['gender'])
        if gender not in [0, 1]:
            errors.append("Gender must be 0 (Female) or 1 (Male).")
        form_data['gender'] = gender

        cp = int(request.form['chest_pain'])
        if cp not in [0, 1, 2, 3]:
            errors.append("Chest pain type must be 0-3.")
        form_data['chest_pain'] = cp

        trest_bps = int(request.form['trest_bps'])
        if not (100 <= trest_bps <= 200):
            errors.append("Resting BP must be between 100 and 200.")
        form_data['trest_bps'] = trest_bps

        chol = int(request.form['cholesterol'])
        if not (125 <= chol <= 565):
            errors.append("Cholesterol must be between 125 and 565.")
        form_data['cholesterol'] = chol

        fbs = int(request.form['fbs'])
        if fbs not in [0, 1]:
            errors.append("FBS must be 0 or 1.")
        form_data['fbs'] = fbs

        rest_ecg = int(request.form['rest_ecg'])
        if rest_ecg not in [0, 1, 2]:
            errors.append("Rest ECG must be 0-2.")
        form_data['rest_ecg'] = rest_ecg

        hr = int(request.form['heart_rate'])
        if not (70 <= hr <= 200):
            errors.append("Heart rate must be between 70 and 200.")
        form_data['heart_rate'] = hr

        exang = int(request.form['exang'])
        if exang not in [0, 1]:
            errors.append("Exang must be 0 or 1.")
        form_data['exang'] = exang

        oldpeak = float(request.form['oldpeak'])
        if not (0.0 <= oldpeak <= 6.0):
            errors.append("Oldpeak must be between 0.0 and 6.0.")
        form_data['oldpeak'] = oldpeak

        slope = int(request.form['slope'])
        if slope not in [0, 1, 2]:
            errors.append("Slope must be 0-2.")
        form_data['slope'] = slope

        ca = int(request.form['ca'])
        if ca not in [0, 1, 2, 3]:
            errors.append("CA must be 0-3.")
        form_data['ca'] = ca

        thal = int(request.form['thal'])
        if thal not in [0, 1, 2, 3]:
            errors.append("Thal must be 0-3.")
        form_data['thal'] = thal

        if errors:
            return render_template("main.html",
                                   errors=errors,
                                   form_data=request.form)

        input_data = np.array([[
            form_data['age'], form_data['gender'], form_data['chest_pain'],
            form_data['trest_bps'], form_data['cholesterol'], form_data['fbs'],
            form_data['rest_ecg'], form_data['heart_rate'], form_data['exang'],
            form_data['oldpeak'], form_data['slope'], form_data['ca'],
            form_data['thal']
        ]])

        prediction = model.predict(input_data)[0]
        result = "High Risk of Heart Attack 💔" if prediction == 1 else "Low Risk of Heart Attack ❤️"
        level = "High Risk" if prediction == 1 else "Low Risk"

        return render_template("result.html",
                               prediction_text=result,
                               risk_level=level,
                               form_data=form_data)

    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=81, debug=True)
