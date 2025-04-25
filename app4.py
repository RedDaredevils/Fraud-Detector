from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and feature columns
model = joblib.load("fraud_model.pkl")  # Make sure the model is saved in this location
model_columns = joblib.load("model_columns.pkl")  # Make sure model_columns is saved in this location

# Root route that renders the form page
@app.route('/')
def home():
    return render_template('index.html')  # Render the form page (index.html)

# Handle form submission and prediction
@app.route('/predict_form', methods=['POST'])
def predict_form():
    try:
        # Collect form data
        data = {
            "step": float(request.form['step']),
            "amount": float(request.form['amount']),
            "oldbalanceOrg": float(request.form['oldbalanceOrg']),
            "newbalanceOrig": float(request.form['newbalanceOrig']),
            "oldbalanceDest": float(request.form['oldbalanceDest']),
            "newbalanceDest": float(request.form['newbalanceDest']),
            "type_CASH_OUT": 0,
            "type_DEBIT": 0,
            "type_PAYMENT": 0,
            "type_TRANSFER": 0
        }

        # Set the selected transaction type
        selected_type = request.form['type']
        data[f"type_{selected_type}"] = 1  # Update the corresponding transaction type to 1

        # Prepare data for prediction
        df = pd.DataFrame([data])
        df = df.reindex(columns=model_columns, fill_value=0)  # Ensure all features are present

        # Make prediction
        prediction = model.predict(df)[0]
        proba = model.predict_proba(df)[0][1]  # Probability of fraud

        # Return prediction and probability to the template
        return render_template("index.html", prediction=int(prediction), probability=round(proba, 4))

    except Exception as e:
        return f"Error: {e}"

# API route for JSON POST requests (optional for testing)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        df = pd.DataFrame([data])
        df = df.reindex(columns=model_columns, fill_value=0)

        # Make prediction and return the result
        prediction = model.predict(df)[0]
        proba = model.predict_proba(df)[0][1]

        return jsonify({
            "prediction": int(prediction),
            "fraud_probability": round(float(proba), 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# Run app locally
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
