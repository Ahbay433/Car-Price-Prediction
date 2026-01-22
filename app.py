from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import numpy as np
import os

app = Flask(__name__)

# ==============================
# Base directory
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==============================
# Load trained ML model
# ==============================
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'car_price_model.pkl')
model = pickle.load(open(MODEL_PATH, 'rb'))

# ==============================
# Load CSV for dropdowns
# ==============================
CSV_PATH = os.path.join(BASE_DIR, 'quikr_car.csv')
car = pd.read_csv(CSV_PATH)

# Basic cleaning for dropdowns
car['company'] = car['company'].astype(str).str.strip()
car['name'] = car['name'].astype(str).str.strip()
car['fuel_type'] = car['fuel_type'].astype(str).str.strip()
car['year'] = pd.to_numeric(car['year'], errors='coerce')

# Remove junk company names
car = car[car['company'].str.len() > 2]

# ==============================
# Home page
# ==============================
@app.route('/')
def index():
    companies = sorted(car['company'].dropna().unique())
    years = sorted(car['year'].dropna().astype(int).unique(), reverse=True)
    fuel_types = sorted(car['fuel_type'].dropna().unique())

    companies.insert(0, 'Select Company')

    return render_template(
        'index.html',
        companies=companies,
        years=years,
        fuel_types=fuel_types
    )

# ==============================
# Dependent dropdown
# ==============================
@app.route('/fetch_models/<company>')
def fetch_models(company):
    company = company.strip()

    models = sorted(
        car[car['company'] == company]['name']
        .dropna()
        .unique()
        .tolist()
    )

    return jsonify(models)

# ==============================
# Prediction route
# ==============================
@app.route('/predict', methods=['POST'])
def predict():
    company = request.form.get('company')
    car_model = request.form.get('car_models')
    year = request.form.get('year')
    fuel_type = request.form.get('fuel_type')
    driven = request.form.get('kilo_driven')

    if not (company and car_model and year and fuel_type and driven):
        return "Please fill all details"

    try:
        year = int(year)
        driven = float(driven)
    except:
        return "Invalid numeric input"

    X = pd.DataFrame(
        [[car_model, company, fuel_type, year, driven]],
        columns=['name', 'company', 'fuel_type', 'year', 'kms_driven']
    )

    prediction = model.predict(X)[0]

    return str(round(prediction, 2))

# ==============================
# Run app (RENDER READY)
# ==============================
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
