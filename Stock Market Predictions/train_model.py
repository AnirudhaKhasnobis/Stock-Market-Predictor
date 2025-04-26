import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

# Load the processed data
df = pd.read_csv('processed_stock_data.csv')

# Drop rows with missing values (if any)
df = df.dropna()

# Features: 50-day Moving Average and RSI
X = df[['50_MA', 'RSI']]

# Target: Close price
y = df['Close']

# Split data into training and testing sets (80% for training, 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Create and train the Linear Regression model
model = LinearRegression()
print("Training the model...")  # Print statement to confirm training
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate Mean Squared Error (MSE) to evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Optionally, save the trained model using joblib
joblib.dump(model, 'stock_price_model.pkl')
print("Model saved as 'stock_price_model.pkl'")
