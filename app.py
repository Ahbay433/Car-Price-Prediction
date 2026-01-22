from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

# ==============================
# Load trained ML model
# ==============================
model = pickle.load(open('model/car_price_model.pkl', 'rb'))

# ==============================
# Load CSV for dropdowns
# ==============================
car = pd.read_csv('quikr_car.csv')

# Basic cleaning for dropdowns
car['company'] = car['company'].astype(str).str.strip()
car['name'] = car['name'].astype(str).str.strip()
car['fuel_type'] = car['fuel_type'].astype(str).str.strip()
car['year'] = pd.to_numeric(car['year'], errors='coerce')

# Remove junk company names (like numbers, very short strings)
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
# Dependent dropdown for car models
# (MATCHES YOUR index.html)
# ==============================
@app.route('/fetch_models/<company>')
def fetch_models(company):
    print(f"DEBUG: fetch_models called with {company}")

    company = company.strip()

    models = sorted(
        car[car['company'] == company]['name']
        .dropna()
        .unique()
        .tolist()
    )

    print(f"DEBUG: models found = {len(models)}")

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

    # Must match model training features
    X = pd.DataFrame(
        [[car_model, company, fuel_type, year, driven]],
        columns=['name', 'company', 'fuel_type', 'year', 'kms_driven']
    )

    prediction = model.predict(X)[0]

    return str(round(prediction, 2))

# ==============================
# Run app
# ==============================
if __name__ == '__main__':
    app.run(debug=True, port=5002)
