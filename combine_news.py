import pandas as pd
import os

data_dir = "data/training_news"

files = [
    "AAPL.csv",
    "AMZN.csv",
    "GOOGL.csv",
    "META.csv",
    "MSFT.csv",
    "NVDA.csv",
    "TSLA.csv",
]

dfs = [pd.read_csv(os.path.join(data_dir, f)) for f in files]

combined = pd.concat(dfs, ignore_index=True)

print("Combined shape:", combined.shape)
print(combined.head())

output_path = os.path.join(data_dir, "labeled_news_all.csv")
combined.to_csv(output_path, index=False)
print(f"\nCombined news saved to: {output_path}")
