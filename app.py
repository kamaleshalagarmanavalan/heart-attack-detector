from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('CARDIO.pkl', 'rb'))

# Store user inputs globally (optional, not needed now)
user_input_data = {}

@app.route('/')
def home():
    return render_template('index.html')  # Welcome page

@app.route('/info')
def info():
    return render_template('info.html')

@app.route('/main')
def main():
    return render_template('main.html')   # Main form page

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get form input values and cast to correct types
            form_data = {
                'age': int(request.form['age']),
                'gender': int(request.form['gender']),
                'chest_pain': int(request.form['chest_pain']),
                'trest_bps': int(request.form['trest_bps']),
                'cholestorol': int(request.form['cholesterol']),
                'fbs': int(request.form['fbs']),
                'rest_ecg': int(request.form['rest_ecg']),
                'heart_rate': int(request.form['heart_rate']),
                'exang': int(request.form['exang']),
                'oldpeak': float(request.form['oldpeak']),
                'slope': int(request.form['slope']),
                'ca': int(request.form['ca']),
                'thal': int(request.form['thal']),
            }

            # Convert values to numpy array for model prediction
            input_data = np.array([[*form_data.values()]])
            prediction = model.predict(input_data)[0]

            # Interpret result
            if prediction == 1:
                result = "High Risk of Heart Attack 💔"
                level = "High Risk"
            else:
                result = "Low Risk of Heart Attack ❤️"
                level = "Low Risk"

            # Pass results to the result page
            return render_template("result.html",
                                   prediction_text=result,
                                   risk_level=level,
                                   form_data=form_data)

        except Exception as e:
            return f"Something went wrong: {str(e)}"
    return "Invalid Request"
    
if __name__ == '__main__':
    app.run(debug=True)
