# Stock Data Collection Project

This repository contains a comprehensive stock data collection system that gathers both price data and news articles for major tech stocks.

## 📊 Data Available

### Stock Price Data
- **7 Major Stocks**: AAPL, NVDA, MSFT, AMZN, GOOGL, META, TSLA
- **Format**: CSV files with daily OHLCV data

### News Data
- **100+ articles per stock** (collected using Marketaux API)
- **Format**: CSV files with structured news data

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd Stock-Predictor
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up API Keys
Create a `.env` file in the project root directory:
```bash
# .env file
STOCK_NEWS_API_TOKEN=your_stocknewsapi_token_here
```

**Where to get API keys:**
- **StockNewsAPI**: Sign up at [https://stocknewsapi.com/](https://stocknewsapi.com/) — a free tier is available, though it may not provide enough request capacity for this project; contact **kta98@sfu.ca** for API key access.

---

## 🚀 Running the Complete Pipeline

### **Step 1: Collect Stock Price Data**

Collects historical OHLCV data from Yahoo Finance for all tickers.

```bash
python 01_collect_data.py
```

**What it does:**
- Downloads stock prices from Yahoo Finance (2023-01-27 to present)
- Saves data to `data/prices/[TICKER].csv`
- No API key required (uses yfinance)

**Configuration:**
- Edit `TICKERS` list in the script to add/remove stocks
- Edit `START` and `END` dates for different time periods
- Default tickers: AAPL, NVDA, MSFT, AMZN, GOOGL, META, TSLA

**Output files:**
```
data/prices/
├── AAPL.csv
├── NVDA.csv
├── MSFT.csv
├── AMZN.csv
├── GOOGL.csv
├── META.csv
└── TSLA.csv
```

---

### **Step 2: Collect News Data**

Collects financial news articles for sentiment analysis.

```bash
python 01_collect_news.py
```

**What it does:**
- Fetches 500 news articles per stock from StockNewsAPI
- Splits data: 400 articles for training, 100 for prediction
- Saves training data to `data/training/news/[TICKER].csv`
- Saves prediction data to `data/predict/news/[TICKER].csv`

**Requirements:**
- Valid `STOCK_NEWS_API_TOKEN` in `.env` file
- Internet connection
- May take several minutes due to API rate limits

**Output files:**
```
data/training/news/
├── AAPL.csv (400 articles)
├── NVDA.csv
├── MSFT.csv
├── AMZN.csv
├── GOOGL.csv
├── META.csv
└── TSLA.csv

data/predict/news/
├── AAPL.csv (100 articles)
├── NVDA.csv
├── MSFT.csv
├── AMZN.csv
├── GOOGL.csv
├── META.csv
└── TSLA.csv
```

---

### **Step 3: Compute Technical Indicators**

Calculates technical indicators and metrics from price data.

```bash
python 03_compute_metrics.py
```

**What it does:**
- Reads raw price data from `data/prices/`
- Calculates technical indicators:
  - SMA (Simple Moving Average)
  - EMA (Exponential Moving Average)
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - Stochastic Oscillators (%K, %D)
  - Volatility metrics
- Saves enriched data to `data/prices_with_metrics/[TICKER].csv`

**Output files:**
```
data/prices_with_metrics/
├── AAPL.csv (prices + technical indicators)
├── NVDA.csv
├── MSFT.csv
├── AMZN.csv
├── GOOGL.csv
├── META.csv
└── TSLA.csv
```

---
