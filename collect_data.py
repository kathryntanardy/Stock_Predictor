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


# TICKERS = ["AAPL", "NVDA", "MSFT", "AMZN", "GOOGL", "META", "TSLA"]
TICKERS = ["TSLA"]
START = "2023-01-27"
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
    all_frames = []

    for t in TICKERS:
        df = yf.download(t, start=START, end=END, interval="1d", auto_adjust=True, progress=False)
        if df.empty:
            print(f"[WARNING] No price data found for {t}")
            continue

        df = df.rename(columns=str.lower).reset_index().rename(columns={"Date": "ds", "date": "ds"})
        df["ticker"] = t
        df = df[["ds", "ticker", "open", "high", "low", "close", "volume"]]
        df.to_csv(PRICES_DIR / f"{t}.csv", index=False)
        all_frames.append(df)
        print(f"[SUCCESS] Saved {t} ({len(df)} rows)")

    if all_frames:
        all_data = pd.concat(all_frames, ignore_index=True)
        all_data.to_csv(PRICES_DIR / "all_prices.csv", index=False)
        print("\n[SUCCESS] Saved combined prices -> data/prices/all_prices.csv")

# collect marketaux news 
def collect_news_marketaux(symbol, start, end):
    key = os.getenv("MARKETAUX_KEY")
    if not key:
        print("[ERROR] MARKETAUX_KEY missing in .env file")
        return pd.DataFrame()

    url = "https://api.marketaux.com/v1/news/all"
    rows, page = [], 1
   
    if start is None:
        start = START  
    
    start_date = datetime.strptime(start, "%Y-%m-%d")
    end_date = datetime.now()

    news_start = start_date.strftime("%Y-%m-%d")
    news_end = end_date.strftime("%Y-%m-%d")
    
    print(f"[INFO] Collecting news for {symbol} from {news_start} to {news_end}")
    print(f"[DEBUG] API Key: {key[:10]}...{key[-5:] if len(key) > 15 else 'SHORT'}")

    # Split the date range into smaller chunks to get more articles
    current_end = datetime.strptime(news_end, "%Y-%m-%d")
    chunk_days = 30  # 30-day chunks
    
    while len(rows) < MAX_ARTICLES_PER_STOCK:
        chunk_start = current_end - timedelta(days=chunk_days)
        chunk_start_str = chunk_start.strftime("%Y-%m-%d")
        chunk_end_str = current_end.strftime("%Y-%m-%d")
        
        print(f"[INFO] Collecting from {chunk_start_str} to {chunk_end_str}")
        
        params = {
            "api_token": key,
            "symbols": symbol,
            "published_after": chunk_start_str,
            "published_before": chunk_end_str,
            "limit": ARTICLES_PER_REQUEST,
            "language": "en"
        }
        
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        response_data = r.json()
        batch = response_data.get("data", [])
        
        print(f"[DEBUG] API Response for {symbol}: {len(batch)} articles from {chunk_start_str} to {chunk_end_str}")
        
        if not batch:
            print(f"[DEBUG] No more articles found for {symbol} in this date range")
            current_end = chunk_start
            if current_end <= datetime.strptime(news_start, "%Y-%m-%d"):
                print(f"[DEBUG] Reached start date, stopping collection")
                break
            continue
            
        rows.extend(batch)
        print(f"[INFO] Collected {len(rows)} articles for {symbol} so far")
        
        # Move to previous chunk
        current_end = chunk_start
        if current_end <= datetime.strptime(news_start, "%Y-%m-%d"):
            print(f"[DEBUG] Reached start date, stopping collection")
            break
            
        time.sleep(1) 

    df = pd.json_normalize(rows)
    if df.empty:
        return df

    df = pd.DataFrame({
        "published_at": pd.to_datetime(df.get("published_at"), errors="coerce"),
        "ticker": symbol,
        "title": df.get("title"),
        "desc": df.get("description"),
        "url": df.get("url"),
        "source": df.get("source"),
        "sentiment": df.get("overall_sentiment"),
        "sentiment_score": df.get("overall_sentiment_score")
    }).drop_duplicates(subset=["url"]).dropna(subset=["published_at"])

    return df


def collect_news():
    print("\n[INFO] Collecting news articles...\n")
    all_frames = []

    # Basic search queries
    # queries = {
    #     "AAPL": "Apple OR AAPL",
    #     "NVDA": "Nvidia OR NVDA",
    #     "MSFT": "Microsoft OR MSFT",
    #     "AMZN": "Amazon OR AMZN",
    #     "GOOGL": "Alphabet OR Google OR GOOGL",
    #     "META": "Meta OR Facebook OR META",
    #     "TSLA": "Tesla OR TSLA"
    # }
    queries = {
        "TSLA": "Tesla OR TSLA"
    }

    # Additional search terms for more comprehensive news collection
    # additional_queries = {
    #     "AAPL": ["Apple Inc", "iPhone", "iPad", "MacBook", "Apple Watch", "Tim Cook"],
    #     "NVDA": ["NVIDIA", "GPU", "AI chips", "Jensen Huang", "RTX", "CUDA"],
    #     "MSFT": ["Microsoft", "Azure", "Office 365", "Windows", "Satya Nadella", "Xbox"],
    #     "AMZN": ["Amazon", "AWS", "Jeff Bezos", "Andy Jassy", "Prime", "Alexa"],
    #     "GOOGL": ["Google", "Alphabet", "Sundar Pichai", "YouTube", "Android", "Chrome"],
    #     "META": ["Meta", "Facebook", "Mark Zuckerberg", "Instagram", "WhatsApp", "VR"],
    #     "TSLA": ["Tesla", "Elon Musk", "Model S", "Model 3", "Model Y", "Cybertruck"]
    # }

    additional_queries = {
       "TSLA": ["Tesla", "Elon Musk", "Model S", "Model 3", "Model Y", "Cybertruck"]
    }

    for t in TICKERS:
        print(f"\n[INFO] Collecting news for {t}...")

        if df.empty:
            print(f"[WARNING] No news found for {t}")
            continue

        if df.empty: 
            print(f"[WARNING] No news found for {t}")
            continue
        else:
            df = collect_news_marketaux(t, START, END)

        df.to_csv(NEWS_DIR / f"{t}_news.csv", index=False)
        all_frames.append(df)
        print(f"[SUCCESS] Saved {t} ({len(df)} articles)")

    if all_frames:
        all_news = pd.concat(all_frames, ignore_index=True)
        all_news.to_csv(NEWS_DIR / "all_news.csv", index=False)
        print(f"\n[SUCCESS] Saved combined news -> data/news/all_news.csv ({len(all_news)} total articles)")


if __name__ == "__main__":
    print("[INFO] Starting data collection script...")
    try:
        collect_prices()
        collect_news()
        print("\n[SUCCESS] Data collection completed successfully!\n")
    except Exception as e:
        print(f"[ERROR] Error occurred: {e}")
        import traceback
        traceback.print_exc()