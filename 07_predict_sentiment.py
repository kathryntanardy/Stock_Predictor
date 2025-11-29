import os
import glob
from datetime import timedelta
import pandas as pd
import joblib

MODEL_PATH = "sentiment_model.joblib"
INPUT_DIR = "data/predict/news"
OUTPUT_DIR = "data/predicted_news_with_sentiment"
TEXT_COL = "text"
DATE_COL = "date"


def print_last_100d_summary(df, ticker_name, date_col=DATE_COL, sentiment_col="sentiment"):
    if date_col not in df.columns:
        print(f"{ticker_name}: no '{date_col}' column found.")
        return

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col].astype(str), utc=True, errors="coerce")
    df = df.dropna(subset=[date_col])

    if df.empty:
        print(f"{ticker_name}: no valid dates.")
        return

    df = df.sort_values(date_col)
    latest = df[date_col].max()
    cutoff = latest - timedelta(days=100)
    df_100 = df[df[date_col] >= cutoff]

    neg = (df_100[sentiment_col] == "Negative").sum()
    neu = (df_100[sentiment_col] == "Neutral").sum()
    pos = (df_100[sentiment_col] == "Positive").sum()
    total = neg + neu + pos

    if total > 0:
        sentiment_score = (pos - neg) / total
    else:
        sentiment_score = 0

    print(f"{ticker_name} – last 100 days: {neg} neg, {neu} neu, {pos} pos, score = {sentiment_score:.3f}")


def main():
    print("Loading model:", MODEL_PATH)
    model = joblib.load(MODEL_PATH)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.csv")))
    if not files:
        print("No CSV files found in", INPUT_DIR)
        return

    print(f"Found {len(files)} file(s)")

    has_proba = hasattr(model, "predict_proba")

    for path in files:
        print("\nProcessing:", path)
        df = pd.read_csv(path)

        if TEXT_COL not in df.columns:
            print(f"  Column '{TEXT_COL}' not found. Skipping.")
            continue

        ticker_name = os.path.splitext(os.path.basename(path))[0]
        texts = df[TEXT_COL].astype(str).fillna("")
        df["sentiment"] = model.predict(texts)

        if has_proba:
            proba = model.predict_proba(texts)
            for i, cls in enumerate(model.classes_):
                df[cls] = proba[:, i]

        print_last_100d_summary(df, ticker_name=ticker_name)

        out_path = os.path.join(OUTPUT_DIR, os.path.basename(path))
        df.to_csv(out_path, index=False)
        print("  -> Saved to:", out_path)

if __name__ == "__main__":
    main()