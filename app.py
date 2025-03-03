import pandas as pd
import numpy as np
import streamlit as st
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load dataset from CSV
data = pd.read_csv("house_prices.csv")

# Splitting dataset
X = data[["Size", "Bedrooms"]]
y = data["Price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model
with open("linear_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Streamlit UI
st.title("House Price Prediction")
size = st.number_input("Enter house size (sq ft):")
bedrooms = st.number_input("Enter number of bedrooms:")

if st.button("Predict"):
    with open("linear_model.pkl", "rb") as f:
        model = pickle.load(f)
    prediction = model.predict(np.array([[size, bedrooms]]))
    st.success(f"Predicted Price: ${prediction[0]:,.2f}")
