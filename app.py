import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# --------------------
# PAGE CONFIG
# --------------------

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# --------------------
# LOAD DATA
# --------------------

data = pd.read_csv("train.csv")

X = data[['GrLivArea', 'BedroomAbvGr', 'FullBath']]
y = data['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# --------------------
# HEADER
# --------------------

st.title("🏠 House Price Prediction Dashboard")

st.markdown(
    "Predict house prices using **Linear Regression** based on square footage, bedrooms, and bathrooms."
)

# --------------------
# KPI CARDS
# --------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("R² Score", f"{r2:.3f}")

with col2:
    st.metric("MAE", f"${mae:,.0f}")

with col3:
    st.metric("Records", len(data))

# --------------------
# SIDEBAR
# --------------------

st.sidebar.header("House Details")

sqft = st.sidebar.slider(
    "Square Footage",
    500,
    5000,
    2000
)

bedrooms = st.sidebar.slider(
    "Bedrooms",
    1,
    10,
    3
)

bathrooms = st.sidebar.slider(
    "Bathrooms",
    1,
    5,
    2
)

# --------------------
# PREDICTION
# --------------------

input_data = pd.DataFrame({
    "GrLivArea": [sqft],
    "BedroomAbvGr": [bedrooms],
    "FullBath": [bathrooms]
})

price = model.predict(input_data)[0]

st.subheader("Predicted Price")

st.success(f"Estimated House Price: ${price:,.2f}")

# --------------------
# ACTUAL VS PREDICTED
# --------------------

chart_data = pd.DataFrame({
    "Actual": y_test,
    "Predicted": predictions
})

fig = px.scatter(
    chart_data,
    x="Actual",
    y="Predicted",
    title="Actual vs Predicted House Prices"
)

st.plotly_chart(fig, use_container_width=True)

# --------------------
# DATASET PREVIEW
# --------------------

st.subheader("Dataset Preview")

st.dataframe(data.head(20))

# --------------------
# REAL HOUSE TEST
# --------------------

st.subheader("Sample Prediction From Dataset")

sample = X_test.iloc[[0]]

actual_price = y_test.iloc[0]
pred_price = model.predict(sample)[0]

col1, col2 = st.columns(2)

with col1:
    st.metric("Actual Price", f"${actual_price:,.0f}")

with col2:
    st.metric("Predicted Price", f"${pred_price:,.0f}")