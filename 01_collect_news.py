import requests
import json 
import os
import pandas as pd
from dotenv import load_dotenv


load_dotenv()
TICKERS = ["AAPL", "NVDA", "MSFT", "AMZN", "GOOGL", "META", "TSLA"]
API_TOKEN = os.getenv("STOCK_NEWS_API_TOKEN")


def get_stock_news(ticker):
    for i in range (1, 6):

        params = {
            "tickers":{ticker},
            "items": 100,
            "page": 1,
            "date": "01282023-10142025",  # 2023-01-28 to 2025-10-14 MMDDYYYY
            "token": API_TOKEN
        }
        
        # response = requests.get("https://stocknewsapi.com/api/v1/", params=params)
        response = requests.get(f"https://stocknewsapi.com/api/v1?tickers={ticker}&items=100&page={i}&token={API_TOKEN}")
        
        response.raise_for_status()

        data =response.json().get("data", [])

        output_directory = "data/training/news"
        os.makedirs(output_directory, exist_ok=True)

        df = pd.DataFrame(data)
        df = df.drop(columns=["image_url", "type"], errors="ignore")
        df["tickers"]=f"{ticker}"
        csv_file = os.path.join(output_directory, f"{ticker}.csv")
        
        if os.path.exists(csv_file):
            df.to_csv(csv_file, mode="a", header=False, index=False)
            print(f"Appended to existing {ticker}.csv")
        else:
            df.to_csv(csv_file, mode="w", header=True, index=False)
            print(f"Created new {ticker}.csv file")
    
def split_training_and_predicting(ticker):
    
    input = f"data/training/news/{ticker}.csv"
    df = pd.read_csv(input)

    first_100 = df.head(100)
    first_100 = first_100.drop(columns=["sentiment"], errors="ignore")
    last_400 = df.tail(400)

    predicting_directory = "data/predict/news"
    os.makedirs(predicting_directory, exist_ok=True)
    first_100.to_csv(f"{predicting_directory}/{ticker}.csv", index=False)

    last_400.to_csv(input, index=False)
    print(f"{ticker}: Separated training (400 entries) and Predicting (100 entries) data")

def main():

    for ticker in TICKERS:
        get_stock_news(ticker)
        split_training_and_predicting(ticker)
    


if __name__ == "__main__":
    main()
    

