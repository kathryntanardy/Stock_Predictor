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

**Step 4: Train Forecasting Models (Price & Direction)**

Trains the machine learning models for both:
- **Classification**: next-day and 7-day UP/DOWN direction  
- **Regression**: next-day and 7-day predicted closing prices  

```bash
# python 04_training_model.py
```

**What it does:**

🔹 Classification Pipeline
- Loads merged price + technical indicator dataset
- Trains multiple classification models using TimeSeriesSplit
- Evaluates each model using: Accuracy,Precision,Recall,F1 Score (primary metric)
- Selects the best classifier based on highest average F1 score
- Generates:Next-day direction prediction (UP or DOWN),7-day multi-horizon direction predictions, BUY/HOLD or SELL/HOLD signals

🔹 Regression Pipeline
- Trains multiple regression models:
- Evaluates each model using:
- MAE (Mean Absolute Error) — primary metric
- Selects the best regressor based on lowest average MAE
- Generates: Next-day predicted closing price,7-day multi-horizon price forecasts
- Computes:Directional accuracy across 7-day horizons

Global average MAE across all horizons
**Output files:**
```
data/model_output/
├── classification_metrics.csv        # CV metrics for all classifiers
├── regression_metrics.csv            # CV metrics for all regressors
├── regression_fs_predictions.csv     # Full-series regression predictions
├── regression_7day_forecasts.csv     # 7-day price forecasts
└── (console output)
```
---

### **Step 5: Combine Training News Data**

Combines all individual stock news files into a single labeled dataset for model training.

```bash
python 05_combine_news.py
```

**What it does:**
- Reads all individual stock news CSV files from `data/training/news/`
- Combines them into a single dataset
- Saves the combined file as `data/training/news/labelled_news_all.csv`

**Requirements:**
- Must have completed Step 2 (news collection)
- All individual stock news files must exist in `data/training/news/`

**Output file:**
```
data/training/news/
└── labelled_news_all.csv (combined dataset from all stocks)
```

---

### **Step 6: Train Sentiment Classification Model**

Trains a sentiment classification model using TF-IDF vectorization and Logistic Regression.

```bash
python 06_train_sentiment_model.py --input-csv data/training/news/labelled_news_all.csv
```

**Or with all options explicitly:**
```bash
python 06_train_sentiment_model.py --input-csv data/training/news/labelled_news_all.csv --text-column text --label-column sentiment --output-model sentiment_model.joblib --cv-folds 10
```

**What it does:**
- Loads labeled news data from the combined CSV file
- Performs cross-validation with F1_macro scoring
- Trains a sentiment classification model using:
  - TF-IDF vectorization (max 20,000 features, 1-2 ngrams)
  - Logistic Regression with balanced class weights
- Evaluates the model on a held-out test set
- Saves the trained model to `sentiment_model.joblib`

**Arguments:**
- `--input-csv` (required): Path to the labeled training data CSV
- `--text-column` (default: "text"): Name of the text column
- `--label-column` (default: "sentiment"): Name of the sentiment label column
- `--output-model` (default: "sentiment_model.joblib"): Path to save the trained model
- `--test-size` (default: 0.2): Fraction of data for testing
- `--random-state` (default: 42): Random seed for reproducibility
- `--cv-folds` (default: 5): Number of cross-validation folds (set 0 or 1 to disable)

**Example with custom options:**
```bash
python 06_train_sentiment_model.py \
    --input-csv data/training/news/labelled_news_all.csv \
    --text-column text \
    --label-column sentiment \
    --output-model sentiment_model.joblib \
    --test-size 0.2 \
    --cv-folds 5
```

**Output:**
- Model file: `sentiment_model.joblib`
- Console output includes:
  - Cross-validation scores (F1_macro per fold and mean)
  - Classification report (precision, recall, F1-score)
  - Confusion matrix

**Requirements:**
- Must have completed Step 5 (combined news data)
- CSV file must contain `text` and `sentiment` columns (or specified column names)

---

### **Step 7: Predict Sentiment on News Data**

Uses the trained sentiment model to predict sentiment scores on new news articles.

```bash
python 07_predict_sentiment.py
```

**What it does:**
- Loads the trained sentiment model from `sentiment_model.joblib`
- Reads news CSV files from `data/predict/news/`
- Predicts sentiment labels (Positive, Neutral, Negative) for each article
- Optionally includes prediction probabilities for each class
- Calculates and displays a 100-day sentiment summary for each ticker
- Saves results to `data/predicted_news_with_sentiment/[TICKER].csv`

**Configuration:**
The script uses constants defined at the top:
- `MODEL_PATH`: Path to the trained model (default: "sentiment_model.joblib")
- `INPUT_DIR`: Directory containing news files to predict (default: "data/predict/news")
- `OUTPUT_DIR`: Directory to save predictions (default: "data/predicted_news_with_sentiment")
- `TEXT_COL`: Name of the text column (default: "text")
- `DATE_COL`: Name of the date column (default: "date")

**Requirements:**
- Must have completed Step 6 (trained model exists)
- News files must exist in `data/predict/news/` (from Step 2)
- Each CSV file must contain a `text` column

**Output files:**
```
data/predicted_news_with_sentiment/
├── AAPL.csv (original data + sentiment predictions + probabilities)
├── NVDA.csv
├── MSFT.csv
├── AMZN.csv
├── GOOGL.csv
├── META.csv
└── TSLA.csv
```

**Output format:**
Each output CSV includes:
- Original columns from input file
- `sentiment`: Predicted sentiment label (Positive/Neutral/Negative)
- `Positive`, `Neutral`, `Negative`: Probability scores for each class (if model supports it)

**Console output:**
For each ticker, displays:
- Processing status
- 100-day sentiment summary: counts of negative/neutral/positive articles and overall sentiment score
- Save confirmation

