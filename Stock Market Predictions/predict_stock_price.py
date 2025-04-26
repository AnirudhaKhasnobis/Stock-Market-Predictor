import joblib
import pandas as pd

# Load the trained model
model = joblib.load('stock_price_model.pkl')

# Example new data (replace with real data or your own input)
new_data = pd.DataFrame({
    '50_MA': [250],  # Example 50-day moving average value
    'RSI': [60]      # Example RSI value
})

# Make a prediction for the closing price
predicted_price = model.predict(new_data)
print(f"Predicted stock price: {predicted_price[0]}")
