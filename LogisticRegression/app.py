import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ---------------------------------------------------
# PAGE TITLE
# ---------------------------------------------------

st.title("Heart Disease Prediction using Logistic Regression")

# ---------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------

df = pd.read_csv("heart.csv")

# ---------------------------------------------------
# DATA PREPROCESSING
# ---------------------------------------------------

encoder = LabelEncoder()

for col in df.select_dtypes(include="object").columns:
    df[col] = encoder.fit_transform(df[col])

# ---------------------------------------------------
# FEATURES AND TARGET
# ---------------------------------------------------

X = df.drop("target", axis=1)

y = df["target"]

# ---------------------------------------------------
# TRAIN TEST SPLIT
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------------------------------------------
# MODEL TRAINING
# ---------------------------------------------------

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# ---------------------------------------------------
# MODEL ACCURACY
# ---------------------------------------------------

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

st.subheader("Model Accuracy")

st.write(f"Accuracy : {accuracy:.2f}")

# ---------------------------------------------------
# USER INPUTS
# ---------------------------------------------------

st.subheader("Enter Patient Details")

age = st.slider("Age", 20, 80, 40)

chol = st.slider("Cholesterol", 100, 400, 200)

thalach = st.slider("Maximum Heart Rate", 60, 220, 150)

# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------

if st.button("Predict"):

    input_data = np.array([[
        age,        # age
        1,          # sex
        0,          # cp
        120,        # trestbps
        chol,       # chol
        0,          # fbs
        1,          # restecg
        thalach,    # thalach
        0,          # exang
        1.0,        # oldpeak
        1,          # slope
        0,          # ca
        2           # thal
    ]])

    result = model.predict(input_data)

    probability = model.predict_proba(input_data)

    if result[0] == 1:

        st.error("Heart Disease Detected")

    else:

        st.success("No Heart Disease Detected")

    st.write(
        f"Prediction Probability : {np.max(probability)*100:.2f}%"
    )

# ---------------------------------------------------
# LOGISTIC REGRESSION CURVE
# ---------------------------------------------------

st.subheader("Logistic Regression Curve")

# Using Age feature for graph

X_graph = df[["age"]]

y_graph = df["target"]

# Train single feature logistic regression

graph_model = LogisticRegression()

graph_model.fit(X_graph, y_graph)

# Generate smooth curve

x_values = np.linspace(
    df["age"].min(),
    df["age"].max(),
    200
).reshape(-1, 1)

y_values = graph_model.predict_proba(x_values)[:, 1]

# Plotting

fig, ax = plt.subplots(figsize=(8, 5))

# Scatter points

ax.scatter(
    df["age"],
    df["target"]
)

# Logistic curve

ax.plot(
    x_values,
    y_values,
    linewidth=3
)

ax.set_xlabel("Age")

ax.set_ylabel("Probability of Heart Disease")

ax.set_title("Logistic Regression Sigmoid Curve")

st.pyplot(fig)