import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,TimeSeriesSplit
import warnings
warnings.filterwarnings("ignore")

#uses random forest as classifier to reduce over fitting
from sklearn.ensemble import RandomForestClassifier 

#import the following to output a report of the model's performance
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, f1_score, precision_score

#standarize the data
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin

import xgboost as xgb #help's with the random forest classifier




def loading_and_preparingData():
#loading data and coverting it to time series using date time
#since stocks are time dependent for future targets
    all_prices = pd.read_csv('data/prices_with_metrics/all_prices.csv')
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
    all_prices = all_prices.dropna(subset= ['next_close']).reset_index(drop= True)

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

def create_features(data):
    data = data.sort_values(['ticker','ds']).copy()
    #price based features
    data['price_change'] = data['close'].pct_change() #finds the percentage change
    data['volume_change'] = data['volume'].pct_change()
    data['high_low_ratio'] = data['high'] /data['low']
    data['close_open_ratio'] = data['close'] /data['open']

    #technical indicators
    data['sma_close_ratio'] = data['close'] / data['SMA']
    data['ema_close_ratio'] = data['close'] / data['EMA']
    data['rsi_overbought'] = (data['RSI'] >  70).astype(int)
    data['rsi_oversold'] = (data['RSI'] < 30).astype(int)
    data['macd_signal'] = (data['MACD'] > 0).astype(int)

    #lags featrues - previous day values
    for c in ["close","volume","RSI", "MACD", "price_change"]:
        data[f"{c}_lag1"] = data.groupby('ticker')[c].shift(1)

    #rolling stats
    data['volatility_5d'] = data.groupby('ticker')['price_change'].rolling(5).std().reset_index(0, drop= True)
    data['volatility_10d'] = data.groupby('ticker')['price_change'].rolling(10).std().reset_index(0, drop= True)

    return data
def prepare_model_data(data):
    features_columns = [
        #price features
        'open','high','low','close','volume','price_change','volume_change','high_low_ratio', 'close_open_ratio',

        #technical indicators
        'SMA', 'EMA','RSI','MACD','%K','%D', 'sma_close_ratio','ema_close_ratio', 'rsi_overbought', 'rsi_oversold','macd_signal',

        #lagged features
        'close_lag1', 'volume_lag1', 'RSI_lag1', 'MACD_lag1', 'price_change_lag1',

        #volatility
        'volatility_5d', 'volatility_10d',

        #news features
        'sentiment_mean', 'sentiment_std', 'sentiment_count', 'article_count'
    ]
    model_data = data[features_columns + ['target', 'ticker', 'ds']].dropna()
    return model_data, features_columns

class LinearRegThresh(BaseEstimator, ClassifierMixin):
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LinearRegression())
        ])

    def fit(self, X, y):
        self.pipe.fit(X, y.astype(float))
        return self

    def predict(self, X):
        pred_prob = self.pipe.predict(X)
        return (pred_prob >= self.threshold).astype(int)

def train_models(X_train, X_test, y_train, y_test):
    models ={}

    #random forest
    #has 100 trees
    rf= RandomForestClassifier(n_estimators = 100, random_state =42, max_depth= 10)
    rf.fit(X_train, y_train)
    models['random forest'] = rf

    #xgb boost 
    #similar to random forest decision tree but differnet
    xgb_model = xgb.XGBClassifier(n_estimators = 100, random_state =42, max_depth= 10)
    xgb_model.fit(X_train, y_train)
    models['XGBoost'] = xgb_model

    #linear reg
    lr_clsfr = LinearRegThresh(threshold = 0.5)
    lr_clsfr.fit(X_train, y_train)
    models['linear regression'] = lr_clsfr

    #decision tree, depth 6
    dt = DecisionTreeClassifier(max_depth = 6, min_samples_leaf = 5, random_state = 42)
    dt.fit(X_train, y_train)
    models['decision tree'] = dt

    #gradient boosting, rate 0.1
    gb = GradientBoostingClassifier(n_estimators = 200, learning_rate = 0.1, max_depth = 3, random_state = 42)
    gb.fit(X_train, y_train)
    models['gradient boosting'] = gb

    #k-nearest neighbors
    knn = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors = 11, weights = 'distance', p = 2))
    ])
    knn.fit(X_train, y_train)
    models['k-nearest neighbors'] = knn

    return models

def eval_models(models, X_test, y_test, feature_names):
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)

        #calculating metrics
        #calculated using the library functions
        accuracy= accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"\n results:")
        print(f"accuracy: {accuracy:}")
        print(f"precision: {precision:}")
        print(f"recall: {recall:}")
        print(f"f1 score: {f1:}")

        #output classification report
        print("\n classification report:")
        print(classification_report(y_test, y_pred,target_names =['DOWN', 'UP']))

        #creates confusion matrix
        print("\n confusion matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(f"predicted: DOWN UP")
        print(f"actual DOWN: {cm[0,0]} {cm[0,1]}") #0,0 is position of the output on the matrix grid #TN, FP
        print(f"actual UP: {cm[1,0]} {cm[1,1]}") #FN, TP

        results[name] = {
            'model' : model,
            'accuracy' : accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': y_pred,
            'confusion_matrix': cm
        }
    return results

def per_ticker_report(model, X_test, y_test, model_data, test_mask):
    test_meta = model.data.loc[test_mask, ['ticker', 'ds', 'target']].reset_index(drop = True)
    preds = test_meta.copy()

def main():
    print("classification model training")

    #loading data
    all_prices, all_news = loading_and_preparingData()
    all_prices = creating_target_variables(all_prices)
    data = preparing_features(all_prices, all_news)
    data = create_features(data)
    model_data, features_columns = prepare_model_data(data)

    #split features /labels
    X = model_data[features_columns]
    y = model_data['target']

    #splitting the data for training and testing
    split_date = model_data['ds'].quantile(0.8)
    train_mask = model_data['ds'] < split_date
    test_mask = model_data['ds'] >= split_date #last 20% for testing

    #creating the train and test metrics
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    #output models based on training on the train and test metrics
    models = train_models(X_train, X_test, y_train, y_test)
    results = eval_models(models, X_test, y_test, features_columns) #report output has 2 reports since 1 is the xgb boost and the other is random forest


if __name__ == "__main__":

    results = main()