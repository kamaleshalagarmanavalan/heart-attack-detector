from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('CARDIO.pkl', 'rb'))

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
        # Age
        age = request.form['age'].strip()
        if not age.isdigit() or not (0 <= int(age) <= 100):
            errors.append("Age must be between 0 and 100.")
        else:
            form_data['age'] = int(age)

        # Gender
        gender = request.form['gender'].strip()
        if gender not in ['0', '1']:
            errors.append("Gender must be 0 (Female) or 1 (Male).")
        else:
            form_data['gender'] = int(gender)

        # Chest Pain
        cp = request.form['chest_pain'].strip()
        if cp not in ['0', '1', '2', '3']:
            errors.append("Chest pain type must be between 0 and 3.")
        else:
            form_data['chest_pain'] = int(cp)

        # Resting BP
        trest_bps = request.form['trest_bps'].strip()
        if not trest_bps.isdigit() or not (95 <= int(trest_bps) <= 200):
            errors.append("Resting BP must be between 95 and 200.")
        else:
            form_data['trest_bps'] = int(trest_bps)

        # Cholesterol
        chol = request.form['cholesterol'].strip()
        if not chol.isdigit() or not (125 <= int(chol) <= 565):
            errors.append("Cholesterol must be between 125 and 565.")
        else:
            form_data['cholestorol'] = int(chol)

        # FBS
        fbs = request.form['fbs'].strip()
        if fbs not in ['0', '1']:
            errors.append("FBS must be 0 or 1.")
        else:
            form_data['fbs'] = int(fbs)

        # Rest ECG
        rest_ecg = request.form['rest_ecg'].strip()
        if rest_ecg not in ['0', '1', '2']:
            errors.append("Rest ECG must be 0, 1, or 2.")
        else:
            form_data['rest_ecg'] = int(rest_ecg)

        # Heart Rate
        heart_rate = request.form['heart_rate'].strip()
        if not heart_rate.isdigit() or not (70 <= int(heart_rate) <= 200):
            errors.append("Heart rate must be between 70 and 200.")
        else:
            form_data['heart_rate'] = int(heart_rate)

        # Exang
        exang = request.form['exang'].strip()
        if exang not in ['0', '1']:
            errors.append("Exang must be 0 or 1.")
        else:
            form_data['exang'] = int(exang)

        # Oldpeak
        oldpeak_input = request.form['oldpeak'].strip().lower()
        if oldpeak_input == 'nan' or oldpeak_input == '':
            errors.append("Oldpeak cannot be empty or NaN.")
        else:
            try:
                oldpeak_val = float(oldpeak_input)
                if not (0.0 <= oldpeak_val <= 6.5):
                    errors.append("Oldpeak must be between 0.0 and 6.5.")
                else:
                    form_data['oldpeak'] = oldpeak_val
            except ValueError:
                errors.append("Oldpeak must be a valid number.")

        # Slope
        slope = request.form['slope'].strip()
        if slope not in ['0', '1', '2']:
            errors.append("Slope must be 0, 1, or 2.")
        else:
            form_data['slope'] = int(slope)

        # CA
        ca = request.form['ca'].strip()
        if ca not in ['0', '1', '2', '3', '4']:
            errors.append("CA must be between 0 and 4.")
        else:
            form_data['ca'] = int(ca)

        # Thal
        thal = request.form['thal'].strip()
        if thal not in ['1', '2', '3']:
            errors.append("Thal must be 1 (Fixed), 2 (Normal), or 3 (Reversible).")
        else:
            form_data['thal'] = int(thal)

        if errors:
            return render_template("main.html", errors=errors, form_data=request.form)

        # Predict
        input_data = np.array([[*form_data.values()]])
        prediction = model.predict(input_data)[0]

        result = "High Risk of Heart Attack 💔" if prediction == 1 else "Low Risk of Heart Attack ❤️"
        level = "High Risk" if prediction == 1 else "Low Risk"

        return render_template("result.html",
                               prediction_text=result,
                               risk_level=level,
                               form_data=form_data)

    except Exception as e:
        return f"Something went wrong: {str(e)}"
