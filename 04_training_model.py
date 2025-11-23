import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, cross_validate
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
from sklearn.base import BaseEstimator, ClassifierMixin, clone

import xgboost as xgb #help's with the random forest classifier

#used later to give a final prediction output
#either next day movement is UP or DOWN
from pandas.tseries.offsets import BDay #considers business days only


#------------------------------------------------------------------------------------------------------------------------------------------
#regression imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#----------------------------
#all data from csv imports
import glob
import os

#output csv for app 
OUTPUT = "data/model_output"
os.makedirs(OUTPUT, exist_ok=True)

def loading_and_preparingData():
#loading data and coverting it to time series using date time
#since stocks are time dependent for future targets
#load all stocks
    files = glob.glob('data/prices_with_metrics/*.csv')
    dfs = []

    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)

    all_prices = pd.concat(dfs, ignore_index = True)
    all_prices['ds'] = pd.to_datetime(all_prices['ds']) #modify the existing df


    return all_prices

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

    return all_prices


def preparing_features(all_prices):
    # Just return the prices data - no news features needed
    #print(f"Price data shape: {all_prices.shape}") #just showing the dimension of the data. for testing purpose
    return all_prices

def create_features(data):
    data = data.sort_values(['ticker','ds']).copy()
    #price based features
    data['price_change']  = data.groupby('ticker')['close'].pct_change()
    data['volume_change'] = data.groupby('ticker')['volume'].pct_change()
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

#------------------------------------------------------------------------------------------------------------------------------------------
#create multiple targets for more than 1 day prediction
def add_multi_targets(data, max_targets = 7):
    data = data.sort_values(['ticker', 'ds']).copy()
    #loop over the 7 days
    for j in range (1, max_targets +1):
        #group by for each day
        data[f'future_close_{j}'] = data.groupby('ticker')['close'].shift(-j) #for each ticker separely look at j days ahead. negative = future  value. so output will be future_close1 and group by ticker and close
    return data


#------------------------------------------------------------------------------------------------------------------------------------------

def prepare_model_data(data):
    features_columns = [
        #price features
        'open','high','low','close','volume','price_change','volume_change','high_low_ratio', 'close_open_ratio',

        #technical indicators
        'SMA', 'EMA','RSI','MACD','%K','%D', 'sma_close_ratio','ema_close_ratio', 'rsi_overbought', 'rsi_oversold','macd_signal',

        #lagged features
        'close_lag1', 'volume_lag1', 'RSI_lag1', 'MACD_lag1', 'price_change_lag1',

        #volatility
        'volatility_5d', 'volatility_10d'
    ]
    model_data = data[features_columns + ['target', 'ticker', 'ds', 'next_close']].dropna() #next_close is for regression
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

def get_models():
    #stores all the different models in the tuple to be used by the cross validation
    #tscv and cross validation requires tuple for validating multiple models
    models ={}

    #random forest
    
    models['random forest']= RandomForestClassifier(n_estimators = 150, random_state =50, max_depth= 6)

    #xgb boost 
    #similar to random forest decision tree but differnet
    models['XGBoost']= xgb.XGBClassifier(n_estimators = 150, random_state =50, max_depth= 6)

    #linear reg
    models['linear regression']  = LinearRegThresh(threshold = 0.5)

    #decision tree, depth 6
    models['decision tree'] = DecisionTreeClassifier(max_depth = 10, min_samples_leaf = 8, random_state = 50)

    #gradient boosting, rate 0.1
    models['gradient boosting'] = GradientBoostingClassifier(n_estimators = 300, learning_rate = 0.1, max_depth = 5, random_state = 50)

    #k-nearest neighbors
    models['k-nearest neighbors'] = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors = 15, weights = 'distance', p = 2))
    ])

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
        #print("\n classification report:")
        #print(classification_report(y_test, y_pred,target_names =['DOWN', 'UP']))

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
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#get regression models
#same process as classification but for regression
def get_regression_models():
    models = {}

    # random forest regressor
    models['rf_reg'] = RandomForestRegressor(
        n_estimators=150,
        random_state=50,
        max_depth=6
    )

    # xgb regressor
    models['xgb_reg'] = xgb.XGBRegressor(
        n_estimators=150,
        random_state=50,
        max_depth=6
    )

    # linear regression
    models['lin_reg'] = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LinearRegression())
    ])

    # decision tree regressor
    models['dt_reg'] = DecisionTreeRegressor(
        max_depth=10,
        min_samples_leaf=8,
        random_state=50
    )

    # gradient boosting regressor
    models['gbr_reg'] = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=5,
        random_state=50
    )

    # knn regressor
    models['knn_reg'] = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsRegressor(
            n_neighbors=15,
            weights='distance',
            p=2
        ))
    ])

    return models
#------------------------------------------------------------------------------------------------------------------------------------------
#helper for the 7 day prediction
def train_multi_target_regressors(base_model, data, features_columns, tscv, n_splits =10, max_targets = 7):
    results = {}
    #for each day, train 1 regression model
    for j in range(1, max_targets +1):
        target_col = f'future_close_{j}'
        #has to modify compared to single day. this is because 7 day has more NaN values in the dataset
        valid = data.dropna(subset= features_columns + [target_col]).reset_index(drop = True)

        #training like the others
        X = valid[features_columns]
        y_h = valid[target_col]

        #folds for regression like the previous
        fold_mae, fold_rmse, fold_r2 = [], [], []

        
        for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train_fold = X.iloc[train_idx]
            X_test_fold = X.iloc[test_idx]
            y_train_fold = y_h.iloc[train_idx]
            y_test_fold = y_h.iloc[test_idx]

            fold_model = clone(base_model)
            fold_model.fit(X_train_fold, y_train_fold)
            y_pred = fold_model.predict(X_test_fold)

            mae = mean_absolute_error(y_test_fold, y_pred)
            mse = mean_squared_error(y_test_fold, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test_fold, y_pred)

            fold_mae.append(mae)
            fold_rmse.append(rmse)
            fold_r2.append(r2)


        avg_mae = np.mean(fold_mae)
        avg_rmse = np.mean(fold_rmse)
        avg_r2 = np.mean(fold_r2)

        # train final model for this day
        final_model = clone(base_model)
        final_model.fit(X, y_h)

        results[j] = {
            'model': final_model,
            'avg_mae': avg_mae,
            'avg_rmse': avg_rmse,
            'avg_r2': avg_r2
        }

    return results

#------------------------------------------------------------------------------------------------------------------------------------------

#same process as classification model eval, but use mse and other regression factors
def eval_regression_models(models, X, y_price, tscv, n_splits):
    all_results = {}

    for name, model in models.items():
        #print("\n")
        #print(f"Evaluating regression model {name}")
        #print("\n")

        fold_mae = []
        fold_rmse = []
        fold_r2 = []

        for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train_fold = X.iloc[train_idx]
            X_test_fold = X.iloc[test_idx]

            y_train_fold = y_price.iloc[train_idx]
            y_test_fold = y_price.iloc[test_idx]

            fold_model = clone(model)
            fold_model.fit(X_train_fold, y_train_fold)

            y_pred_price = fold_model.predict(X_test_fold)

            mae = mean_absolute_error(y_test_fold, y_pred_price)
            mse = mean_squared_error(y_test_fold, y_pred_price)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test_fold, y_pred_price)

            fold_mae.append(mae)
            fold_rmse.append(rmse)
            fold_r2.append(r2)

            #print(f"  Fold {fold_idx+1}: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")

        avg_mae = np.mean(fold_mae)
        std_mae = np.std(fold_mae)
        avg_rmse = np.mean(fold_rmse)
        std_rmse = np.std(fold_rmse)
        avg_r2 = np.mean(fold_r2)
        std_r2 = np.std(fold_r2)

        #print(f"\nAverage regression metrics across {n_splits} folds:")
        #print(f"  MAE:   {avg_mae:.4f}")
        #print(f"  RMSE:  {avg_rmse:.4f}")
        #print(f"  R2:    {avg_r2:.4f}")

        final_model = clone(model)
        final_model.fit(X, y_price)

        all_results[name] = {
            'model': final_model,
            'avg_mae': avg_mae,
            'std_mae': std_mae,
            'avg_rmse': avg_rmse,
            'std_rmse': std_rmse,
            'avg_r2': avg_r2,
            'std_r2': std_r2,
        }
    return all_results

def per_ticker_report(model, X_test, y_test, model_data, test_mask):
    test_meta = model.data.loc[test_mask, ['ticker', 'ds', 'target']].reset_index(drop = True)
    preds = test_meta.copy()

def main():
    print("classification model training with cross-validation")

    #loading data
    all_prices = loading_and_preparingData()
    all_prices = creating_target_variables(all_prices)
    data = preparing_features(all_prices)
    data = create_features(data)

    #for the 7 day
    data = add_multi_targets(data, max_targets =7)
    

    model_data, features_columns = prepare_model_data(data)

    #split features /labels
    X = model_data[features_columns]
    y = model_data['target']

    #cross validation set up
    n_splits = 10 #number of splits can be modified and changes later
    #time series split form of cross validation using k fold
    #= tscv time series cross validator part of the requirement of time series
    tscv = TimeSeriesSplit(n_splits = n_splits) #takes the n_spltis from above

    model_data = model_data.sort_values('ds').reset_index(drop= True)
    X = model_data[features_columns]
    y = model_data['target'] #re-extract x and y after sorting. for classification
    y_price = model_data['next_close'] #for regression
    current_close= model_data['close'] #for classification coverting current price to up/down

    #scoring metrics
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1'
    }


#------------------------------------------------------------------------------------------------------------------
#classification models
    # Get untrained model instances
    models = get_models()

    print(f"\nStarting {n_splits}-fold time series cross-validation...")
    print(f"Total samples: {len(X)}")
    print(f"Target distribution: {y.value_counts().to_dict()}\n")

    all_results = {}
    #output is in the form
    #1  #number of up days
    #0  #number of down days

    # Evaluate each model using cross-validation
    #page breaking between different models
    for name, model in models.items():
        #print("\n")
        #print(f"Evaluating {name}")
        #print("\n")

        #cross validation
        cv_results = cross_validate(
            model, X, y,
            cv=tscv,
            scoring=scoring,
            return_train_score=False,
            n_jobs=1 #how much of the cpu is used
            #if is -1, 100% cpu power is used
        )

        # collect predictions for confusion matrix and classification report
        y_pred = np.full(len(y), -1, dtype=int)  # Use -1 to mark unpredicted samples
        all_y_true = []
        all_y_pred = []
        
        for train_idx, test_idx in tscv.split(X):
            X_train_fold = X.iloc[train_idx]
            X_test_fold = X.iloc[test_idx]
            y_train_fold = y.iloc[train_idx]
            y_test_fold = y.iloc[test_idx]
            
            # Creating a fresh model instance for this fold
            fold_model = clone(model)
            fold_model.fit(X_train_fold, y_train_fold)
            fold_predictions = fold_model.predict(X_test_fold)
            y_pred[test_idx] = fold_predictions
            
            # Collect for aggregated metrics
            all_y_true.extend(y_test_fold.values)
            all_y_pred.extend(fold_predictions)

        # Calculate average metrics across folds
        avg_accuracy = np.mean(cv_results['test_accuracy'])
        std_accuracy = np.std(cv_results['test_accuracy'])
        avg_precision = np.mean(cv_results['test_precision'])
        std_precision = np.std(cv_results['test_precision'])
        avg_recall = np.mean(cv_results['test_recall'])
        std_recall = np.std(cv_results['test_recall'])
        avg_f1 = np.mean(cv_results['test_f1'])
        std_f1 = np.std(cv_results['test_f1'])

        # Overall metrics from aggregated predictions (only from test folds)
        overall_accuracy = accuracy_score(all_y_true, all_y_pred)
        overall_precision = precision_score(all_y_true, all_y_pred, zero_division=0)
        overall_recall = recall_score(all_y_true, all_y_pred, zero_division=0)
        overall_f1 = f1_score(all_y_true, all_y_pred, zero_division=0)
        overall_cm = confusion_matrix(all_y_true, all_y_pred)

        # PRINTS PER FOLD RESULT FOR TESTING PURPOSES
        #print(f"\nPer-fold results:")
       #for fold_idx in range(n_splits):
            #print(f"  Fold {fold_idx+1}:")
            #print(f"    Accuracy:  {cv_results['test_accuracy'][fold_idx]:.4f}")
            #print(f"    Precision: {cv_results['test_precision'][fold_idx]:.4f}")
            #print(f"    Recall:    {cv_results['test_recall'][fold_idx]:.4f}")
            #print(f"    F1 Score:  {cv_results['test_f1'][fold_idx]:.4f}")

        # Print summary
        #print(f"\nAverage across {n_splits} folds:")
        #print(f"  Accuracy:  {avg_accuracy:.4f}")
        #print(f"  Precision: {avg_precision:.4f}")
        #print(f"  Recall:    {avg_recall:.4f}")
        #print(f"  F1 Score:  {avg_f1:.4f}")

        #print(f"\nOverall metrics:")
        #print(f"  Accuracy:  {overall_accuracy:.4f}")
        #print(f"  Precision: {overall_precision:.4f}")
        #print(f"  Recall:    {overall_recall:.4f}")
        #print(f"  F1 Score:  {overall_f1:.4f}")

        #print(f"\nConfusion Matrix:")
        #print(f"  Predicted: DOWN UP")
        #print(f"  Actual DOWN: {overall_cm[0,0]} {overall_cm[0,1]}")
        #print(f"  Actual UP:   {overall_cm[1,0]} {overall_cm[1,1]}")

       #print(f"\nClassification Report:")
        #print(classification_report(all_y_true, all_y_pred, target_names=['DOWN', 'UP']))
        
        # Train final model on all data for potential future use
        final_model = model
        final_model.fit(X, y)

        all_results[name] = {
            'model': final_model,
            'avg_accuracy': avg_accuracy,
            'avg_precision': avg_precision,
            'avg_recall': avg_recall,
            'avg_f1_score': avg_f1,
            'std_accuracy': std_accuracy,
            'std_precision': std_precision,
            'std_recall': std_recall,
            'std_f1_score': std_f1,
            'overall_accuracy': overall_accuracy,
            'overall_precision': overall_precision,
            'overall_recall': overall_recall,
            'overall_f1_score': overall_f1,
            'confusion_matrix': overall_cm,
            'cv_results': cv_results
        }
    


    #from all the models: pick the best result
    #picks the model with the best f1 score
    best_name = max(all_results, key= lambda k: all_results[k]['avg_f1_score']) #finds best f1score
    best_model = all_results[best_name]['model'] #finds the label of the best model
    
    #latest data
    latest = (model_data.sort_values(['ticker','ds']).groupby('ticker', as_index = False).tail(1))

    #latest prediction
    X_latest = latest[features_columns]
    predic = best_model.predict(X_latest)

    #probability of it being up

    #signals table - used to format output
    #just using the previous variables but putting it in a new table
    signals = latest[['ticker','ds']].rename(columns = {'ds': 'as_of_date'}).copy()
    signals['model'] = best_name
    signals['pred'] = predic

    #signal actions
    signals['label'] = signals['pred'].map({1: 'UP', 0:'DOWN'})
    signals['action'] = signals['pred'].map({1: 'SELL', 0:'BUY/HOLD'}) #if its down, then buy or hold
                                                                        #if its up then signal to sell the stock
                                                                        #signals sell regardless of profit margin

    signals['predicted_for'] = signals['as_of_date'] + BDay(1) #signals the price for the next day. but only considers business day


    #PRINTING FINAL PREDICTION OUTPUT FOR SINGLE DAY
    print("\nBest model by CV and overall F1:", best_name)
    #print("\n Nextday perticker signals")
    cols_to_show = ['ticker', 'as_of_date', 'predicted_for', 'pred', 'label', 'action']
    #print(signals[cols_to_show].to_string(index=False))


    #+---------------save classification metrics (output) for app---------------+
    class_rows = []

    for model_name, info in all_results.items():
        class_rows.append({
            'model': model_name,
            'avg_accuracy': info['avg_accuracy'],
            'std_accuracy': info['std_accuracy'],
            'overall_accuracy': info['overall_accuracy'],
            'avg_precision': info['avg_precision'],
            'std_precision': info['std_precision'],
            'overall_precision': info['overall_precision'],
            'avg_recall': info['avg_recall'],
            'std_recall': info['std_recall'],
            'overall_recall': info['overall_recall'],
            'avg_f1_score': info['avg_f1_score'],
            'std_f1_score': info['std_f1_score'],
            'overall_f1_score': info['overall_f1_score']
        })

    class_metrics_df = pd.DataFrame(class_rows)
    class_metrics_path = os.path.join(OUTPUT, "classification_metrics.csv")
    class_metrics_df.to_csv(class_metrics_path, index=False)
    print(f"\nClassification metrics saved to {class_metrics_path}")
    #+--------------------------------------------------------------------------+


#------------------------------------------------------------------------------------------------------------------
#regression models
    reg_models = get_regression_models()
    reg_results = eval_regression_models(reg_models, X, y_price, tscv, n_splits)

#pick best regression model from least MSE
    best_reg_name = min(reg_results, key = lambda k: reg_results[k]['avg_mae'])
    best_reg_model = reg_results[best_reg_name]['model']


    price_pred = best_reg_model.predict(X_latest)
    latest_close = latest['close'].values

#similar to way clssification was implemented
    reg_signals = latest[['ticker', 'ds']].rename(columns={'ds': 'as_of_date'}).copy()
    reg_signals['model'] = best_reg_name
    reg_signals['predicted_price'] = price_pred
    reg_signals['direction'] = np.where(price_pred > latest_close, 'UP', 'DOWN')
    reg_signals['action'] = np.where(price_pred > latest_close, 'SELL', 'BUY/HOLD')
    reg_signals['predicted_for'] = reg_signals['as_of_date'] + BDay(1)

    print("\nBest regression model by CV MAE:", best_reg_name)
    #print("\n Nextday per-ticker regression signals")
    cols_to_show_reg = [
        'ticker', 'as_of_date', 'predicted_for',
        'predicted_price', 'direction', 'action'
    ]
    #print(reg_signals[cols_to_show_reg].to_string(index=False))

    
    #+---------------save regression metrics (output) for app---------------+
    reg_rows = []

    for model_name, info in reg_results.items():
        reg_rows.append({
            "model": model_name,
            "avg_mae": info['avg_mae'],
            "std_mae": info['std_mae'],
            "avg_rmse": info['avg_rmse'],
            "std_rmse": info['std_rmse'],
            "avg_r2": info['avg_r2'],
            "std_r2": info['std_r2']
        })

    reg_metrics_df = pd.DataFrame(reg_rows)
    reg_metrics_path = os.path.join(OUTPUT, "regression_metrics.csv")
    reg_metrics_df.to_csv(reg_metrics_path, index=False)
    print(f"\nRegression metrics saved to {reg_metrics_path}")
    #+--------------------------------------------------------------------------+

    #save full-series regression predictions for all dates
    all_price_pred = best_reg_model.predict(X)

    reg_pred_df = model_data[['ticker', 'ds', 'close', 'next_close']].copy()
    reg_pred_df['predicted_next_close'] = all_price_pred

    reg_pred_path = os.path.join(OUTPUT, "regression_fs_predictions.csv")
    reg_pred_df.to_csv(reg_pred_path, index=False)
    print(f"\nFull-series regression predictions saved to {reg_pred_path}")


#------------------------------------------------------------------------------------------------------------------
#7 day output training
    #from the 1 day regression prediction above, it already picks the best model for it
    #using that model, the prediction of 7 from it will be based on that
    #but 7 day prediction is not predicting based on the previous day that was also a prediction
    base_reg_model = best_reg_model #takes best reg model from earlier
    multi_reg_results = train_multi_target_regressors(base_model = base_reg_model, data= data, features_columns = features_columns, tscv= tscv, n_splits = n_splits, max_targets =7 )
#------------------------------------------------------------------------------------------------------------------
#directional accuracy for regression
    print('regression accuracy for 7 day model')
    #initialization of variables
    total_mae_error = 0.0 #float
    total_accurate = 0
    total_count = 0
    for j in range(1,8):
        if j not in multi_reg_results:
            continue
        target_col = f'future_close_{j}'
        #original data + future close data
        recent_data = data.dropna(subset = features_columns + [target_col]).sort_values(['ticker', 'ds']).reset_index(drop = True)

        X_h = recent_data[features_columns]
        y_true_h = recent_data[target_col].values #price at time of future close
        y_ref_h = recent_data['close'].values #current price at closing

        model_h = multi_reg_results[j]['model']
        y_pred_h = model_h.predict(X_h)

        #directional accuracy of regression
        #considers accuracy in predicting up or down
        true_dir = (y_true_h > y_ref_h).astype(int)
        pred_dir = (y_pred_h > y_ref_h).astype(int)
        
        #how far prediction is from the actual price
        #using mae
        total_mae_error +=  np.abs(y_true_h - y_pred_h).sum() #accumulate mae across 7 da7s

        #sum of accuracy across 7 days for average accuracy across 7 days
        #how accurate it is going in that direction. if its UP or DOWN
        total_accurate += (true_dir == pred_dir).sum()
        total_count += len(y_true_h)

    overall_mae = (total_mae_error / total_count) 
    overall_accuracy_reg = (total_accurate / total_count) * 100
    

    print("\nregression accuracy")
    print(f"how far off from prediction(avg difference (predicted close - real close) across 7 day prediction): {overall_mae} ")
    #accuracy across all stocks
    print(f"directional accuracy (avg accuracy across 7 day prediction): {overall_accuracy_reg}")



#------------------------------------------------------------------------------------------------------------------


    #outputting prediction of the 7 days
    X_latest_multi = latest[features_columns]

    multi_signals = []
    latest_dates = latest['ds'].values
    latest_tickers = latest['ticker'].values
    latest_close = latest['close'].values

    for j in range(1, 8):  #in range of 1-7 but its 8 cuz of the indices 
        if j not in multi_reg_results:
            continue

        model_j = multi_reg_results[j]['model']
        price_pred_j = model_j.predict(X_latest_multi)

        for i in range(len(latest)):
            multi_signals.append({
                'ticker': latest_tickers[i],
                'as_of_date': latest_dates[i],
                '+target_days': j,
                'predicted_for': latest_dates[i] + BDay(j),
                'latest_close': latest_close[i],
                'predicted_price': price_pred_j[i],
                'direction': 'UP' if price_pred_j[i] > latest_close[i] else 'DOWN',
                'action': 'SELL' if price_pred_j[i] > latest_close[i] else 'BUY/HOLD'
            })

    multi_signals_df = pd.DataFrame(multi_signals)

    print("\n7-day per-ticker regression forecasts:")
    print(multi_signals_df
          .sort_values(['ticker', 'as_of_date', '+target_days'])
          .to_string(index=False))

    #return both results
    return {
        'classification': all_results, #all_results was used from classification
        'regression': reg_results, #return regression
        '7_day_regression': multi_reg_results # return 7 day reg
    }

if __name__ == "__main__":

    results = main()

#note: rmse and r2 are not used in 7 day prediction and only for testing purposes intially (t help figure out the models)
#also arent use in 7 day as best model is linear regression and combining with market volataility doesnt help with accuracy metric