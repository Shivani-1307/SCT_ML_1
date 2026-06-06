import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
data = pd.read_csv("train.csv")

# Show first rows
print(data.head())

# Select important columns
X = data[['GrLivArea', 'BedroomAbvGr', 'FullBath']]

# Target column
y = data['SalePrice']

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nModel Evaluation")
print("Mean Absolute Error:", mae)
print("R2 Score:", r2)

# Test on a real house from dataset
sample_house = X_test.iloc[[0]]

predicted_price = model.predict(sample_house)

print("\n===== Real House Prediction =====")
print("House Features:")
print(sample_house)

print(f"\nActual Price    : ${y_test.iloc[0]:,.2f}")
print(f"Predicted Price : ${predicted_price[0]:,.2f}")

# User Input Prediction
print("\n===== Predict Your Own House =====")

sqft = float(input("Enter square footage: "))
bedrooms = int(input("Enter number of bedrooms: "))
bathrooms = int(input("Enter number of bathrooms: "))

new_house = [[sqft, bedrooms, bathrooms]]

user_prediction = model.predict(new_house)

print(f"\nEstimated House Price: ${user_prediction[0]:,.2f}")