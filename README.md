# Stock Price Prediction using LSTM

## Overview
This project aims to predict stock prices using historical stock data and a Long Short-Term Memory (LSTM) model. The dataset is obtained from Yahoo Finance using the `yfinance` library, and the model is trained on past stock prices to forecast future closing prices.

## Features
- Fetch historical stock data using Yahoo Finance.
- Preprocess data by normalizing and transforming it.
- Implement an LSTM model for time series prediction.
- Train the model on stock price data.
- Evaluate and visualize the predicted stock prices.

## Dataset
- The dataset consists of historical stock prices of **Apple Inc. (AAPL)** from **2014-06-08 to 2025-02-06**.
- The key column used for prediction is the `Close` price.

## Dependencies
Ensure you have the following libraries installed before running the project:
```bash
pip install yfinance numpy pandas matplotlib scikit-learn tensorflow keras
```

## Implementation Steps
### 1. Fetching Stock Data
A function is created to fetch stock data from Yahoo Finance using `yfinance`. The data is then stored in a Pandas DataFrame.
```python
import yfinance as yf

def stock_data(stock, start_date, end_date):
    tickerSymbol = stock
    tickerData = yf.Ticker(tickerSymbol)
    tickerDf = tickerData.history(period='1d', start=start_date, end=end_date)
    return tickerDf

df = stock_data("AAPL", "2014-06-08", "2025-02-06")
df.head()
```

### 2. Data Preprocessing
- The dataset is checked for data types and descriptive statistics.
- Unnecessary columns such as `Dividends` and `Stock Splits` are removed.
- The `Close` price is used for training.
```python
df.drop(columns=['Dividends', 'Stock Splits'], inplace=True)
df1 = df[['Close']].copy()
```

### 3. Train-Test Split
- The data is split into **training** (1750 samples) and **testing** (remaining samples).
```python
data = df1.values
train = data[0:1750, :]
test = data[1750:, :]
```

### 4. Data Normalization
- The data is normalized to the range **0-1** using `MinMaxScaler`.
```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)
```

### 5. Preparing Training & Testing Data
- Data is transformed into sequences of 60 days for LSTM input.
```python
x_train, y_train = [], []
window = 60
for i in range(window, len(train)):
    x_train.append(scaled_data[i-window:i, 0])
    y_train.append(scaled_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
```

### 6. Building the LSTM Model
- A sequential LSTM model is built with **two LSTM layers** and a **Dense output layer**.
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()
```

### 7. Training the Model
- The model is trained for **2 epochs** with a batch size of **64**.
```python
model.fit(x_train, y_train, epochs=2, batch_size=64, verbose=2)
```

### 8. Making Predictions
- Predictions are made using the trained model.
- The predicted values are inverse-transformed back to their original scale.
```python
closing_price = model.predict(x_test)
closing_price = scaler.inverse_transform(closing_price)
```

### 9. Performance Evaluation
- **Root Mean Squared Error (RMSE)** is calculated to evaluate the model.
```python
from sklearn.metrics import mean_squared_error
import math

mse = math.sqrt(mean_squared_error(test, closing_price))
print(f"Root Mean Squared Error: {mse}")
```

### 10. Visualization
- The actual and predicted stock prices are plotted.
```python
import matplotlib.pyplot as plt

test['Predictions'] = closing_price
plt.figure(figsize=(17,8))
plt.xlabel("Year")
plt.ylabel("Closing Price in USD")
plt.title("CLOSING PRICE PREDICTION")
plt.plot(train['Close'])
plt.plot(test[['Close', 'Predictions']])
plt.legend(["Train", "Actual Price", "Predicted Price"])
plt.show()
```

## Results
- The model provides reasonable stock price predictions.
- Achieving high accuracy requires **more extensive data, fine-tuning, and additional features**.

## Future Improvements
- Include additional features like **trading volume, technical indicators, and news sentiment**.
- Experiment with **different LSTM architectures and hyperparameters**.
- Increase training epochs for better convergence.

## Conclusion
This project demonstrates a basic approach to stock price prediction using LSTMs. While useful, financial markets are highly complex, and more sophisticated models are needed for accurate forecasting.
