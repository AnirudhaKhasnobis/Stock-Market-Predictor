import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Load the processed data and trained model
df = pd.read_csv('processed_stock_data.csv')
model = joblib.load('stock_price_model.pkl')

# Prepare features and target as done during training
X = df[['50_MA', 'RSI']]
y = df['Close']

# Split the data for prediction (using the same split used during training)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Plotting actual vs predicted stock prices
plt.figure(figsize=(10,6))
plt.plot(y_test.index, y_test, label='Actual', color='blue')  # Actual prices
plt.plot(y_test.index, y_pred, label='Predicted', color='red')  # Predicted prices
plt.title('Actual vs Predicted Stock Prices')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
