"""
What is the most complicated code you have written independently without AI or anyone else's assistance? 
Attach the question as multiline comment above your code and upload the file.
"""
from flask import Flask, render_template, request, redirect, url_for, jsonify
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import io
import base64
import os
import joblib
import requests
from datetime import datetime, timedelta

app = Flask(__name__)

# API Key for Alpha Vantage
API_KEY = '5KYZ6YDIWW25BZBG'  # Replace with your API key

# Popular stock symbols
STOCK_SYMBOLS = {
    'AAPL': 'Apple Inc.',
    'MSFT': 'Microsoft Corporation',
    'AMZN': 'Amazon.com, Inc.',
    'GOOGL': 'Alphabet Inc.',
    'META': 'Meta Platforms, Inc.',
    'TSLA': 'Tesla, Inc.',
    'NVDA': 'NVIDIA Corporation',
    'JPM': 'JPMorgan Chase & Co.',
    'V': 'Visa Inc.',
    'JNJ': 'Johnson & Johnson'
}

@app.route('/')
def index():
    return render_template('index.html', stocks=STOCK_SYMBOLS)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the selected stock symbol from the form
    symbol = request.form.get('stock_symbol')
    
    if not symbol:
        return render_template('index.html', error="Please select a stock symbol", stocks=STOCK_SYMBOLS)
    
    try:
        # Fetch stock data
        stock_data = fetch_stock_data(symbol)
        
        if stock_data is None:
            return render_template('index.html', error="Failed to fetch stock data. Please try again later.", stocks=STOCK_SYMBOLS)
        
        # Preprocess data
        processed_data = preprocess_data(stock_data)
        
        # Train model (or use existing model)
        model_file = f'models/{symbol}_model.pkl'
        if os.path.exists(model_file):
            model = joblib.load(model_file)
        else:
            # Create directory if it doesn't exist
            os.makedirs('models', exist_ok=True)
            model = train_model(processed_data)
            joblib.dump(model, model_file)
        
        # Make prediction for tomorrow
        latest_data = processed_data.iloc[-1]
        prediction_input = pd.DataFrame({
            '50_MA': [latest_data['50_MA']],
            'RSI': [latest_data['RSI']]
        })
        
        predicted_price = model.predict(prediction_input)[0]
        
        # Get visualization
        plot_url = create_visualization(processed_data, model)
        
        # Get latest price and calculate change
        latest_price = processed_data['Close'].iloc[-1]
        price_change = predicted_price - latest_price
        percent_change = (price_change / latest_price) * 100
        
        return render_template(
            'result.html',
            stock_name=STOCK_SYMBOLS.get(symbol, symbol),
            stock_symbol=symbol,
            latest_price=round(latest_price, 2),
            predicted_price=round(predicted_price, 2),
            price_change=round(price_change, 2),
            percent_change=round(percent_change, 2),
            prediction_date=(datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'),
            plot_url=plot_url
        )
    
    except Exception as e:
        return render_template('index.html', error=f"An error occurred: {str(e)}", stocks=STOCK_SYMBOLS)

def fetch_stock_data(symbol):
    """Fetch stock data from Alpha Vantage API"""
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize=full&apikey={API_KEY}'
    
    response = requests.get(url)
    data = response.json()
    
    if "Time Series (Daily)" in data:
        time_series = data["Time Series (Daily)"]
        dates = list(time_series.keys())
        prices = []
        
        for date in dates:
            prices.append({
                'Open': float(time_series[date]['1. open']),
                'High': float(time_series[date]['2. high']),
                'Low': float(time_series[date]['3. low']),
                'Close': float(time_series[date]['4. close']),
                'Volume': float(time_series[date]['5. volume'])
            })
        
        df = pd.DataFrame(prices, index=dates)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()  # Ensure dates are in chronological order
        
        return df
    else:
        return None

def preprocess_data(df):
    """Preprocess the stock data"""
    # Calculate 50-day moving average
    df['50_MA'] = df['Close'].rolling(window=50).mean()
    
    # Calculate Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Drop NaN values
    df = df.dropna()
    
    return df

def train_model(df):
    """Train a linear regression model"""
    from sklearn.linear_model import LinearRegression
    
    # Features and target
    X = df[['50_MA', 'RSI']]
    y = df['Close']
    
    # Create and train model
    model = LinearRegression()
    model.fit(X, y)
    
    return model

def create_visualization(df, model):
    """Create visualization of actual vs predicted prices"""
    # Make predictions for the entire dataset
    X = df[['50_MA', 'RSI']]
    y_actual = df['Close']
    y_pred = model.predict(X)
    
    # Create plot
    plt.figure(figsize=(12, 6))
    plt.plot(df.index[-100:], y_actual[-100:], label='Actual', color='blue')
    plt.plot(df.index[-100:], y_pred[-100:], label='Predicted', color='red')
    plt.title('Actual vs Predicted Stock Prices (Last 100 Days)')
    plt.xlabel('Date')
    plt.ylabel('Stock Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot to a buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    
    # Convert plot to base64 string
    plot_data = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return f"data:image/png;base64,{plot_data}"

if __name__ == '__main__':
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)

    app.run(debug=True)
