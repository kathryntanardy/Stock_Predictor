import pandas as pd
import numpy as np
import os

input_dir = "data/prices"
output_dir = "data/prices_with_metrics"
os.makedirs(output_dir, exist_ok=True)


def compute_indicators(price_data):
    #SMA (Window can be changed to 10 or 20)
    price_data['SMA'] = price_data['close'].rolling(window=10).mean()

    #EMA (Span can be modified)
    price_data['EMA'] = price_data['close'].ewm(span=20, adjust=False).mean()

    #RSI
    length = 14
    diff = price_data['close'].diff()
    gain = np.where(diff > 0, diff, 0)
    loss = np.where(diff < 0, -diff, 0)
    gain_avg = pd.Series(gain).rolling(window=length, min_periods=1).mean()
    loss_avg = pd.Series(loss).rolling(window=length, min_periods=1).mean()
    rs = gain_avg/loss_avg
    price_data['RSI'] = 100 - (100 /(1 + rs))

    #MACD (EMA 12, 26)
    ema_12 = price_data['close'].ewm(span=12, adjust=False).mean()
    ema_26 = price_data['close'].ewm(span=26, adjust=False).mean()
    price_data['MACD'] = ema_12 - ema_26

    #Stoch
    low_14 = price_data['low'].rolling(window=14).min()
    high_14 = price_data['high'].rolling(window=14).max()
    price_data['%K'] = 100 * ((price_data['close'] - low_14)/(high_14 - low_14))
    price_data['%D'] = price_data['%K'].rolling(window=3).mean()

    return price_data


for file in os.listdir(input_dir):
    if file.endswith(".csv"):
        print(f"Processing {file}")

        file_path = os.path.join(input_dir, file)
        data = pd.read_csv(file_path, parse_dates=['ds'])
        data = compute_indicators(data)

        output_path = os.path.join(output_dir, file)
        data.to_csv(output_path, index=True)

