import requests
import json 
import os
import pandas as pd
from dotenv import load_dotenv


load_dotenv()
TICKERS = ["AAPL", "NVDA", "MSFT", "AMZN", "GOOGL", "META", "TSLA"]
API_TOKEN = os.getenv("STOCK_NEWS_API_TOKEN")


def get_stock_news(ticker):

    params = {
        "tickers":{ticker},
        "items": 100,
        "page": 1,
        "date": "01282023-10142025",  # 2023-01-28 to 2025-10-14 MMDDYYYY
        "token": API_TOKEN
    }
    
    # response = requests.get("https://stocknewsapi.com/api/v1/", params=params)
    response = requests.get(f"https://stocknewsapi.com/api/v1?tickers={ticker}&items=100&page=1&token={API_TOKEN}")
    
    response.raise_for_status()

    data =response.json().get("data", [])

    output_directory = "data/news"
    os.makedirs(output_directory, exist_ok=True)

    df = pd.DataFrame(data)
    df = df.drop(columns=["sentiment", "image_url", "type"], errors="ignore")
    df["tickers"]=f"{ticker}"
    csv_file = os.path.join(output_directory, f"{ticker}.csv")
    df.to_csv(csv_file, index=False)


    print(f"Saved {len(df)} news articles for {ticker} to '{csv_file}'")

def get_stock_training_news(ticker):
    
    response = requests.get(f"https://stocknewsapi.com/api/v1?tickers={ticker}&items=100&page=1&token={API_TOKEN}")    
    response.raise_for_status()

    data =response.json().get("data", [])

    output_directory = "data/training_news"
    os.makedirs(output_directory, exist_ok=True)

    df = pd.DataFrame(data)
    df = df.drop(columns=["image_url", "type"], errors="ignore")
    df["tickers"]=f"{ticker}"
    csv_file = os.path.join(output_directory, f"{ticker}.csv")
    df.to_csv(csv_file, index=False)


    print(f"Saved {len(df)} training news articles for {ticker} to '{csv_file}'")


def main():

    for ticker in TICKERS:
        get_stock_news(ticker)
        get_stock_training_news(ticker)
    


if __name__ == "__main__":
    main()
    

