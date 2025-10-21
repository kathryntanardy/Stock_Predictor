# Stock Data Collection Project

This repository contains a comprehensive stock data collection system that gathers both price data and news articles for major tech stocks.

## 📊 Data Available

### Stock Price Data
- **7 Major Stocks**: AAPL, NVDA, MSFT, AMZN, GOOGL, META, TSLA
- **Time Period**: 2020 to present (1,455+ days per stock)
- **Data Points**: Open, High, Low, Close, Volume
- **Format**: CSV files with daily OHLCV data

### News Data
- **100+ articles per stock** (collected using Marketaux API)
- **Time Period**: Last 3 years of news
- **Data Points**: Title, Description, URL, Source, Sentiment, Date
- **Format**: CSV files with structured news data

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd Data_Collection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Data Collection (Optional)
```bash
python collect_data.py
```

## 📁 File Structure

```
Data_Collection/
├── collect_data.py          # Main data collection script
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── .gitignore              # Git ignore rules
├── data/
│   ├── prices/             # Stock price data
│   │   ├── AAPL.csv       # Apple stock data
│   │   ├── NVDA.csv       # NVIDIA stock data
│   │   ├── MSFT.csv       # Microsoft stock data
│   │   ├── AMZN.csv       # Amazon stock data
│   │   ├── GOOGL.csv      # Google stock data
│   │   ├── META.csv       # Meta stock data
│   │   ├── TSLA.csv       # Tesla stock data
│   │   └── all_prices.csv # Combined price data
│   └── news/               # News article data
│       ├── AAPL_news.csv  # Apple news articles
│       ├── NVDA_news.csv  # NVIDIA news articles
│       ├── MSFT_news.csv  # Microsoft news articles
│       ├── AMZN_news.csv  # Amazon news articles
│       ├── GOOGL_news.csv # Google news articles
│       ├── META_news.csv  # Meta news articles
│       ├── TSLA_news.csv  # Tesla news articles
│       └── all_news.csv   # Combined news data
```

## 🔧 For Group Members

### Using the Data
The data is already collected and ready to use! You can:

1. **Load stock prices**:
   ```python
   import pandas as pd
   df = pd.read_csv('data/prices/AAPL.csv')
   print(df.head())
   ```

2. **Load news data**:
   ```python
   import pandas as pd
   news = pd.read_csv('data/news/AAPL_news.csv')
   print(news.head())
   ```

3. **Load combined data**:
   ```python
   import pandas as pd
   all_prices = pd.read_csv('data/prices/all_prices.csv')
   all_news = pd.read_csv('data/news/all_news.csv')
   ```

### Collecting New Data
If you want to collect fresh data:

1. **Get a Marketaux API key** (free): https://marketaux.com/
2. **Create a `.env` file**:
   ```
   MARKETAUX_KEY=your_api_key_here
   ```
3. **Run the collection script**:
   ```bash
   python collect_data.py
   ```

## 📈 Data Details

### Stock Price Data Format
| Column | Description |
|--------|-------------|
| ds | Date (YYYY-MM-DD) |
| ticker | Stock symbol |
| open | Opening price |
| high | Highest price |
| low | Lowest price |
| close | Closing price |
| volume | Trading volume |

### News Data Format
| Column | Description |
|--------|-------------|
| published_at | Publication date/time |
| ticker | Stock symbol |
| title | Article title |
| desc | Article description |
| url | Article URL |
| source | News source |
| sentiment | Sentiment analysis |
| sentiment_score | Sentiment score |

## 🤝 Contributing

1. Fork the repository
2. Make your changes
3. Submit a pull request

## 📝 Notes

- The `.env` file is excluded from the repository for security
- News collection requires a Marketaux API key
- Stock price data is collected from Yahoo Finance (no API key needed)
- The script respects API rate limits and includes proper error handling
