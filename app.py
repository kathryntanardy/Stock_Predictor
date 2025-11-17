import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(
    page_title="Stock Prediction Dashboard", 
    layout="wide"
)

#load data: regression, classification, sentiment
def load_reg_predictions():
    df = pd.read_csv(
        "data/model_output/regression_fs_predictions.csv", 
        parse_dates=['ds']
    )
    return df

def load_class_metrics():
    df = pd.read_csv("data/model_output/classification_metrics.csv")
    return df

def load_sentiment(ticker):
    path = f"data/predicted_news_with_sentiment/{ticker}.csv"

    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        return None, None
    
    date_col = "date"
    if date_col not in df.columns:
        return df, None

    df[date_col] = pd.to_datetime(df[date_col].astype(str), errors="coerce")
    df = df.dropna(subset=[date_col])

    return df, date_col


#load data
reg_preds = load_reg_predictions()
class_metrics = load_class_metrics()


#sidebar
st.sidebar.markdown("## Navigation")
page = st.sidebar.radio("Prediction Models", ["Main", "Sentiment"])

st.sidebar.markdown("## Select Stock")
stock_tickers = sorted(reg_preds["ticker"].unique())
stock = st.sidebar.selectbox("Ticker", stock_tickers)

df_stock = reg_preds[reg_preds["ticker"] == stock].copy()
if df_stock.empty:
    st.error("No data available for {stock}.")
    st.stop()

min_date = df_stock["ds"].min().date()
max_date = df_stock["ds"].max().date()

st.sidebar.markdown("## Select Date Range")
initial_start = max_date.replace(day = 1)
initial_end = max_date

start_date, end_date = st.sidebar.date_input(
    "Select Date Range",
    value = (initial_start, initial_end),
    min_value = min_date,
    max_value = max_date
)


#main page (regression + classification)
def main_page(stock, reg_preds, class_metrics, start_date, end_date):
    st.markdown("# Stock Price Prediction")
    st.markdown(f"### {stock} -- Actual vs. Predicted Close Prices")

    df = reg_preds[reg_preds["ticker"] == stock].copy()

    mask = (df["ds"].dt.date >= start_date) & (df['ds'].dt.date <= end_date)
    subset = df.loc[mask].sort_values("ds")

    if subset.empty:
        st.warning(f"No data available for {stock} in the selected date range.")
        return
    
    fig = go.Figure()

    fig.add_scatter(
        x = subset["ds"],
        y = subset["close"],
        mode = "lines+markers",
        name = "Actual Close Price",
        line = dict(color = "blue"),
    )

    if "predicted_next_close" in subset.columns:
        fig.add_scatter(
            x = subset["ds"],
            y = subset["predicted_next_close"],
            mode = "lines+markers",
            name = "Predicted Next Close Price",
            line = dict(color = "orange"),
        )    

    fig.update_layout(
        xaxis_title = "Date",
        yaxis_title = "Price (USD)",
        hovermode = "x unified",
    )

    st.plotly_chart(fig, use_container_width = True)

    #classification metrics
    st.markdown("# Classification Metric Assessment")
    if class_metrics is not None and not class_metrics.empty:
        if "avg_accuracy" in class_metrics.columns:
            best_model = class_metrics.loc[class_metrics['avg_accuracy'].idxmax()]
        else:
            best_model = class_metrics.loc[class_metrics['avg_f1_score'].idxmax()]

        col1, col2, col3 = st.columns(3)
        col1.metric("Best Model", best_model['model'])
        col2.metric("Accuracy", f"{best_model['avg_accuracy']:.2%}")


#sentiment page
def sentiment_page(stock):
    st.markdown("# News Sentiment Analytics")
    st.markdown(f"### {stock} -- Sentiment Over Time (Daily Average)")

    df, date_col = load_sentiment(stock)

    if df is None:
        st.warning(f"No sentiment data available for {stock}.")
        return

    if date_col is None:
        st.warning(f"No date column identified in sentiment data.")
        st.dataframe(df.head())
        return

    dt_col = "parsed_datetime"
    df[dt_col] = pd.to_datetime(df[date_col].astype(str), errors="coerce")
    df = df.dropna(subset=[dt_col])

    #zero-centered sentiment score
    zero_neg_col = next(
        (c for c in df.columns if "neg" in c.lower()),
        None
    )
    zero_pos_col = next(
        (c for c in df.columns if "pos" in c.lower()),
        None
    )

    if zero_neg_col is not None and zero_pos_col is not None:
        df["sentiment_score"] = (
            df[zero_pos_col].astype(float) - df[zero_neg_col].astype(float)
        )
    elif "sentiment_score" in df.columns:
        df["sentiment_score"] = df["sentiment_score"].astype(float)
    else:
        df["sentiment_score"] = 0.0

    df["date_only"] = df[dt_col].dt.date

    daily_sentiment = (
        df.groupby("date_only")["sentiment_score"]
        .mean()
        .reset_index()
    )

    if daily_sentiment.empty:
        st.warning(f"No valid sentiment scores available for {stock}.")
        return

    fig = go.Figure(
        data = [
            go.Bar(
                x = daily_sentiment["date_only"],
                y = daily_sentiment["sentiment_score"],
                name = "Average Daily Sentiment Score",
                marker_color = "blue",
            )
        ]
    )

    fig.update_layout(
        xaxis_title = "Date",
        yaxis_title = "Sentiment Score",
        yaxis_zeroline = True,
        yaxis_zerolinewidth = 2,
    )
    
    st.plotly_chart(fig, use_container_width = True)

    #sentiment data table
    st.markdown("# News Sentiment Score")
    
    df["date"] = df[dt_col].dt.date
    df["time"] = df[dt_col].dt.time

    headline_col = None
    for candidate in ["headline", "title"]:
        if candidate in df.columns:
            headline_col = candidate
            break
            
    desc_col = None
    for candidate in ["summary", "description", "snippet"]:
        if candidate in df.columns:
            desc_col = candidate
            break
            
    neg_col = None
    neu_col = None
    pos_col = None

    if "Negative" in df.columns:
        neg_col = "Negative"
    else:
        neg_col = next(
            (c for c in df.columns if "neg" in c.lower()),
            None
        )
    
    if "Neutral" in df.columns:
        neu_col = "Neutral"
    else:
        neu_col = next(
            (c for c in df.columns if "neu" in c.lower() or "neutral" in c.lower()),
            None
        )
                      
    if "Positive" in df.columns:
        pos_col = "Positive"
    else:
        pos_col = next(
            (c for c in df.columns if "pos" in c.lower()),
            None
        )

    cols = ["date", "time"]

    if headline_col is not None:
        cols.append(headline_col)
    if desc_col is not None:
        cols.append(desc_col)
    if neg_col is not None:
        cols.append(neg_col)
    if neu_col is not None:
        cols.append(neu_col)
    if pos_col is not None:
        cols.append(pos_col)

    cols.append("sentiment_score")

    cols = [c for c in cols if c in df.columns]

    #table formatting
    table_df = df[cols].copy()
    table_df = table_df.rename(
        columns = {
            "date": "Date",
            "time": "Time",
            "sentiment_score": "Sentiment Score"
        }
    )
    table_df = table_df.reset_index(drop=True)
    table_df.index += 1  #indexing from 1
    st.dataframe(table_df, use_container_width=True)  
    
    
if page == "Main":
    main_page(stock, reg_preds, class_metrics, start_date, end_date)
else:
    sentiment_page(stock)


