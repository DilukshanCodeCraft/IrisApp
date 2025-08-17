# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.datasets import load_iris
import datetime
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# -------------------------------
# Load trained model
# -------------------------------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Iris Flower Classifier", layout="wide")
st.title("Iris Flower Classification")
st.write("Enter the measurements of the iris flower to predict its species.")

# -------------------------------
# Predefined example flowers
# -------------------------------
species_data = {
    "Iris Setosa": [5.1, 3.5, 1.4, 0.2],
    "Iris Versicolor": [7.0, 3.2, 4.7, 1.4],
    "Iris Virginica": [6.3, 3.3, 6.0, 2.5]
}

species_choice = st.selectbox("Select an example flower", list(species_data.keys()))
default_values = species_data[species_choice]

col1, col2 = st.columns(2)
with col1:
    SepalLengthCm = st.number_input("Sepal Length (cm)", 0.0, 10.0, value=default_values[0])
    SepalWidthCm = st.number_input("Sepal Width (cm)", 0.0, 10.0, value=default_values[1])
with col2:
    PetalLengthCm = st.number_input("Petal Length (cm)", 0.0, 10.0, value=default_values[2])
    PetalWidthCm = st.number_input("Petal Width (cm)", 0.0, 10.0, value=default_values[3])

# -------------------------------
# Input validation warnings
# -------------------------------
if not (4.0 <= SepalLengthCm <= 8.0):
    st.warning("Sepal Length is outside typical range (4.0 - 8.0 cm).")
if not (2.0 <= SepalWidthCm <= 4.5):
    st.warning("Sepal Width is outside typical range (2.0 - 4.5 cm).")
if not (1.0 <= PetalLengthCm <= 7.0):
    st.warning("Petal Length is outside typical range (1.0 - 7.0 cm).")
if not (0.1 <= PetalWidthCm <= 2.5):
    st.warning("Petal Width is outside typical range (0.1 - 2.5 cm).")

# -------------------------------
# Make prediction
# -------------------------------
if st.button("Predict"):
    features = pd.DataFrame([{
        "SepalLengthCm": SepalLengthCm,
        "SepalWidthCm": SepalWidthCm,
        "PetalLengthCm": PetalLengthCm,
        "PetalWidthCm": PetalWidthCm
    }])

    # Predict class
    prediction = model.predict(features)[0]

    # Predict probabilities
    prediction_proba = model.predict_proba(features)[0] if hasattr(model, "predict_proba") else [1.0, 0.0, 0.0]
    confidence = np.max(prediction_proba) * 100

    # Display prediction
    st.success(f"Predicted Class: {prediction}")
    st.info(f"Confidence: {confidence:.2f}%")

    # -------------------------------
    # Confidence Gauge
    # -------------------------------
    gauge_color = "green" if confidence >= 80 else "yellow" if confidence >= 50 else "red"
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        title={'text': "Confidence (%)"},
        gauge={'axis': {'range': [0, 100]},
               'bar': {'color': gauge_color},
               'steps': [
                   {'range': [0, 50], 'color': "#ffcccc"},
                   {'range': [50, 80], 'color': "#ffe680"},
                   {'range': [80, 100], 'color': "#ccffcc"}]}
    ))
    fig_gauge.update_layout(width=800, height=300)
    st.plotly_chart(fig_gauge, use_container_width=True)

    # -------------------------------
    # Probability Bar Chart
    # -------------------------------
    class_labels = model.classes_ if hasattr(model, "classes_") else ["Setosa", "Versicolor", "Virginica"]
    proba_df = pd.DataFrame({"Species": class_labels, "Probability": prediction_proba})
    fig_bar = px.bar(proba_df, x="Species", y="Probability", color="Species",
                     text_auto=".2f", color_discrete_sequence=px.colors.qualitative.Set2)
    fig_bar.update_layout(width=800, height=300)
    st.plotly_chart(fig_bar, use_container_width=True)

    # Log predictions
    with open("predictions_log.csv", "a") as f:
        f.write(f"{datetime.datetime.now()},{species_choice},{SepalLengthCm},{SepalWidthCm},{PetalLengthCm},{PetalWidthCm},{prediction},{confidence}\n")

# -------------------------------
# Scatter plot visualization
# -------------------------------
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target

# Map numeric species to names
species_map = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
df['species_name'] = df['species'].map(species_map)

st.subheader("Scatter Plots of Iris Dataset")
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

sns.scatterplot(data=df, x='sepal length (cm)', y='sepal width (cm)',
                hue='species_name', ax=ax[0], palette="Set1")
ax[0].set_title("Sepal Dimensions")

sns.scatterplot(data=df, x='petal length (cm)', y='petal width (cm)',
                hue='species_name', ax=ax[1], palette="Set2")
ax[1].set_title("Petal Dimensions")

st.pyplot(fig)