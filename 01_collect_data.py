import os
import time
import requests
import pandas as pd
import yfinance as yf
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime, timedelta

# load keys from env
if os.path.exists('.env'):
    with open('.env', 'r', encoding='utf-8-sig') as f:
        for line in f:
            line = line.strip()
            if '=' in line and not line.startswith('#'):
                key, value = line.split('=', 1)
                os.environ[key] = value


TICKERS = ["AAPL", "NVDA", "MSFT", "AMZN", "GOOGL", "META", "TSLA"]
START = "2020-01-02"
END = None  

# news collection configuration
MAX_ARTICLES_PER_STOCK = 100 
ARTICLES_PER_REQUEST = 3 

# folder structure
DATA_DIR = Path("data")
PRICES_DIR = DATA_DIR / "prices"
NEWS_DIR = DATA_DIR / "news"
PRICES_DIR.mkdir(parents=True, exist_ok=True)
NEWS_DIR.mkdir(parents=True, exist_ok=True)

# collect stock prices from yfinance
def collect_prices():
    print("\n[INFO] Collecting stock price data...\n")

    for t in TICKERS:
        df = yf.download(t, start=START, end=END, interval="1d", auto_adjust=True, progress=False)
        if df.empty:
            print(f"[WARNING] No price data found for {t}")
            continue

        df = df.rename(columns=str.lower).reset_index().rename(columns={"Date": "ds", "date": "ds"})
        df["ticker"] = t
        df = df[["ds", "ticker", "open", "high", "low", "close", "volume"]]
        df.to_csv(PRICES_DIR / f"{t}.csv", index=False)
        print(f"[SUCCESS] Saved {t} ({len(df)} rows)")



if __name__ == "__main__":
    print("[INFO] Starting data collection script...")
    try:
        collect_prices()
        print("\n[SUCCESS] Stock data collection completed successfully!\n")
    except Exception as e:
        print(f"[ERROR] Error occurred: {e}")
        import traceback
        traceback.print_exc()