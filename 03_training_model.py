import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,TimeSeriesSplit

#uses random forest as classifier to reduce over fitting
from sklearn.ensemble import RandomForestClassifier 

#import the following to output a report of the model's performance
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#standarize the data
from sklearn.preprocessing import StandardScaler

import xgboost as xgb #help's with the random forest classifier




def loading_and_preparingData():
#loading data and coverting it to time series using date time
#since stocks are time dependent for future targets
    all_prices = pd.read_csv('data/prices/all_prices.csv')
    all_prices['ds'] = pd.to_datetime(all_prices['ds']) #modify the existing df

    #news data
    all_news = pd.read_csv('data/news/all_news.csv')
    all_news['published_at'] = pd.to_datetime(all_news['published_at'])

    print(f"Price data shape: {all_prices.shape}")
    print(f"News data shape: {all_news.shape}")

    return all_prices, all_news

def creating_target_variables(all_prices):
    #binary classification target of up or down

    #sorting by ticker and date
    all_prices = all_prices.sort_values(['ticker','ds']).reset_index(drop=True)

    #creating target varible of 1: where next day close > current close
    #creates a column in the df to group by ticker and close and setting them as int
    all_prices['next_close'] = all_prices.groupby('ticker')['close'].shift(-1)

    #setting column target with the condition that the next close is > current close
    all_prices['target'] = (all_prices['next_close'] > all_prices['close']).astype(int)

    #removing lasr row for each ticker, no next day data
    all_prices = all_prices.groupby('ticker').apply(lambda x: x[:-1]).reset_index(drop=True)

    print(f"target distribution: {all_prices['target'].value_counts()}")
    print(f"target percentage: {all_prices['target'].mean()}")
    return all_prices


def preparing_features(all_prices, all_news):
    #convert sentime score from news data to numeric. missing data will be 0
    all_news['sentiment_score'] = pd.to_numeric(all_news['sentiment_score'], errors = 'coerce').fillna(0)

    #creating new column on df for date but without time
    all_news['date'] = all_news['published_at'].dt.date

    #news features
    news_feature =  all_news.groupby(['ticker', 'date']).agg({
        'sentiment_score': ['mean', 'std', 'count'],
        'title': 'count' #count of articles
    }).reset_index()

    news_feature.columns =(['ticker', 'date', 'sentiment_mean', 'sentiment_std', 'sentiment_count', 'article_count'])
    news_feature['date'] = pd.to_datetime(news_feature['date'])

    #missing values in news features
    news_feature['sentiment_std'] = news_feature['sentiment_std'].fillna(0)

    #merging news features and pricces
    all_prices['date'] = all_prices['ds'].dt.date
    all_prices['date'] = pd.to_datetime(all_prices['date'])

    merged_data = all_prices.merge(news_feature, on= ['ticker', 'date'], how ='left') #left join

    all_news_cols = ['sentiment_mean', "sentiment_std",'sentiment_count', 'article_count']
    merged_data[all_news_cols] = merged_data[all_news_cols].fillna(0) #in the merge data df, news cols NA values are filled with 0

    print(f"data after merging with news:{merged_data.shape}")
    return merged_data


if __name__ == "__main__":
    all_prices, all_news = loading_and_preparingData()
    all_prices = creating_target_variables(all_prices)
    merged = preparing_features(all_prices, all_news)
    print(merged.head(20))