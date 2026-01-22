import pandas as pd
import pickle
import os
import re
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

print("Loading data...")

df = pd.read_csv("quikr_car.csv")

print("Initial rows:", len(df))

# ---------------------------
# CLEAN PRICE
# ---------------------------
def clean_price(x):
    if pd.isna(x):
        return None
    x = str(x)
    if "Ask" in x:
        return None
    x = re.sub(r"[^\d]", "", x)
    return float(x) if x != "" else None

df["Price"] = df["Price"].apply(clean_price)

# ---------------------------
# CLEAN KMS
# ---------------------------
def clean_kms(x):
    if pd.isna(x):
        return None
    x = str(x)
    x = re.sub(r"[^\d]", "", x)
    return float(x) if x != "" else None

df["kms_driven"] = df["kms_driven"].apply(clean_kms)

# ---------------------------
# CLEAN YEAR
# ---------------------------
def clean_year(x):
    try:
        return int(x)
    except:
        return None

df["year"] = df["year"].apply(clean_year)

# ---------------------------
# CLEAN FUEL + COMPANY
# ---------------------------
df["fuel_type"] = df["fuel_type"].astype(str)
df["company"] = df["company"].astype(str)

# ---------------------------
# DROP BAD ROWS
# ---------------------------
df = df.dropna(subset=["Price", "kms_driven", "year", "fuel_type", "company"])

print("Rows after cleaning:", len(df))

# ---------------------------
# FEATURES
# ---------------------------
X = df[["company", "fuel_type", "year", "kms_driven"]]
y = df["Price"]

categorical_features = ["company", "fuel_type"]
numeric_features = ["year", "kms_driven"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features)
    ]
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training model...")
model.fit(X_train, y_train)

# ---------------------------
# SAVE MODEL
# ---------------------------
os.makedirs("model", exist_ok=True)

with open("model/car_price_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved to model/car_price_model.pkl")
print("Training rows used:", len(df))
