#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: carlnordahl

Description:
This script implements a machine learning pipeline to predict the three-day
price movement (up/down) of stocks following their earnings release. It
leverages EPS surprise data from Kaggle and market-related features
(momentum, volatility, market return) from yfinance. The final model is a
hyperparameter-tuned Logistic Regression, evaluated using standard
classification metrics and visualisations.

Project Goal:
Can earnings surprise data along with other financial features predict
the price movement in the days following the earnings release?
"""

from collections import defaultdict
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import yfinance as yf

from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, RocCurveDisplay

import joblib

DATA_KAGGLE_RAW = 'assets/Data.csv'
DATA_PROCESSED_PATH = 'assets/processed_data.csv'
MODEL_SAVE_PATH = ''
SCALER_SAVE_PATH = ''

RANDOM_STATE = 10

TARGET_COLUMN = 'Up/Down'

SELECTED_FEATURES = [
    'Surprise scaled',
    'Surprise Direction',
    'Surprise_scaled_x_Volatility',
    'Market Return',
    'Volatility_x_Market_Return',
    'Relative Surprise',
    'Reported EPS',
    'Market_Return_sq']

MODEL_SAVE_DIR = 'models'
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'best_logistic_regression_model.pkl')
SCALER_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'standard_scaler.pkl')

def load_preprocess_raw(filepath):
    
    # read kaggle data and drop rows with incomplete information
    df = pd.read_csv('assets/Data.csv', sep='|')
    df = df.dropna(subset=['EPS Estimate', 'Reported EPS', 'day0', 'day3'])
    df = df.reset_index(drop=True)
    
    # change format of datetime and drop any duplicate rows
    df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d')
    df.drop_duplicates(inplace=True)
    
    # Redo calculations for surprise (some are missing in the raw data)
    surprises = []

    for estimate, actual in zip(df['EPS Estimate'], df['Reported EPS']):
        surprise = (actual - estimate)/np.abs(estimate)
        surprise_pct = surprise * 100
        surprises.append(surprise_pct)

    df['Surprise'] = surprises
    df['Surprise scaled'] = np.sign(df['Surprise']) * np.log1p(np.abs(df['Surprise']))
    
    # put each row in the correct category
    def classify_return(r):
        return 'up' if r >= 0 else 'down'
    
    df['Return'] = (df['day3'] - df['day0']) / df['day0']
    df[TARGET_COLUMN] = df['Return'].apply(classify_return)
    
    return df

def get_yfinance_data(df):
    
    # create cache for download history
    history_cache = {}
    
    # Count how many times each ticker appears
    # when we have seen all the occurences, we'll delete the cache to free up memory
    ticker_counts = df['Symbol'].value_counts().to_dict()
    seen_counts = defaultdict(int)
    
    momentums = []
    volatilities = []
    market_returns = []
    market_volatilities = []
    
    start_date = df['datetime'].min() - pd.DateOffset(days=21)
    end_date = df['datetime'].max() + pd.DateOffset(days=1)
    
    market_history = yf.Ticker('^GSPC').history(start=start_date, end=end_date)
    vix_hist = yf.Ticker('^VIX').history(start=start_date, end=end_date)
    
    for index, row in df.iterrows():
        ticker = row['Symbol']
        date = row['datetime']
        seen_counts[ticker] += 1
        
        if ticker not in history_cache:
            # we have not seen this ticker before, download history
            stock = yf.Ticker(ticker)
            history_cache[ticker] = stock.history(
                start=start_date, end=end_date)
            
        full_history = history_cache[ticker]
        
        if not full_history.empty and len(full_history) >= 9:
            
            full_history.index = full_history.index.tz_localize(None)
            
            # get specific part of history that we want
            history_window = full_history.loc[
                (full_history.index < date) &
                (full_history.index >= date - pd.DateOffset(days=20))]
            
            # check again to make sure we could retrieve enough data
            if len(history_window) >= 9:
                volatility = history_window['Close'].pct_change().dropna().std()
                momentum = (history_window['Close'].iloc[-1] - history_window['Close'].iloc[-8]) / history_window['Close'].iloc[-8]
                
                momentums.append(momentum)
                volatilities.append(volatility)
            else:
                momentums.append(None)
                volatilities.append(None)
            
            # get market data as well, if history is not found, we don't need it
            market_history_row = market_history.loc[str(date)]
            
            # return of sp500
            market_return = (market_history_row['Open'] - market_history_row['Close']) / market_history_row['Open']
            
            # vix index (volatility)
            market_volatility = vix_hist.loc[str(date)]['Close']
            
            market_returns.append(market_return)
            market_volatilities.append(market_volatility)
        
        else:
            # the corresponding rows will be deleted
            volatilities.append(None)
            momentums.append(None)
            market_returns.append(None)
            market_volatilities.append(None)
        
        if seen_counts[ticker] == ticker_counts[ticker]:
            # we have seen all occurences of this ticker, we don't need to keep the history
            del history_cache[ticker]
    
    df['Momentum'] = momentums
    df['Volatility'] = volatilities
    df['Market Return'] = market_returns
    df['Market Volatility'] = market_volatilities
    
    return df

def engineer_features(df):
    df['Relative Surprise'] = df['Surprise'] / (df['Volatility'] * 100)
    
    directions = []
    
    for surp in df['Surprise']:
        if surp > 0:
            directions.append(1)
        else:
            directions.append(0)
    
    df['Surprise Direction'] = directions
    
    # add interaction-features
    df['Surprise_scaled_x_Volatility'] = df['Surprise scaled'] * df['Volatility']
    df['Surprise_scaled_x_Momentum'] = df['Surprise scaled'] * df['Momentum']
    df['Volatility_x_Momentum'] = df['Volatility'] * df['Momentum']
    df['Surprise_scaled_x_Market_Return'] = df['Surprise scaled'] * df['Market Return']
    df['Volatility_x_Market_Return'] = df['Volatility'] * df['Market Return']
    
    # add some polynomial features
    df['Surprise_scaled_sq'] = df['Surprise scaled']**2
    df['Volatility_sq'] = df['Volatility']**2
    df['Momentum_sq'] = df['Momentum']**2
    df['Market_Return_sq'] = df['Market Return']**2
    
    df.dropna(inplace=True)
    
    return df
    
def perform_print_anova(df, feature_cols, target_col):
    X = df[feature_cols]
    y = df[target_col]
    
    f_val, p_val = f_classif(X, y)
    
    anova_results = pd.DataFrame({
        'Feature': X.columns,
        'F value': f_val,
        'p value': p_val}).sort_values(by='F value', ascending=False)
    
    print('Results from anova test')
    print(anova_results)
    
    anova_results.columns = [col.replace('_', r'\_') for col in anova_results.columns]
    def format_sci(x):
        if isinstance(x, (int, float, np.number)):
            return f"${x:.2e}$"
        return x
    
    anova_results = anova_results.applymap(format_sci)
    
    
    latex_table = anova_results.to_latex(index=False, caption="F- and p-values", label="Fpval")
    print(latex_table)
    
    
def train_evaluate_log_reg(X_train_scaled, X_test_scaled, y_train, y_test, features_list):
    
    # conduct a test of many possible parameter combinations
    paramgrid = {
        'solver' : ['saga', 'liblinear'],
        'max_iter' : [1000, 2000],
        'penalty' : ['l1', 'l2'],
        'C' : np.logspace(-3, 5, 9),
        'class_weight' : [None, 'balanced']
        }
    
    model = LogisticRegression(random_state=RANDOM_STATE)
    
    grid_search = GridSearchCV(estimator=model, param_grid=paramgrid, cv=5, scoring='accuracy', verbose=2)
    
    grid_search.fit(X_train_scaled, y_train)
    
    print(f"Best Logistic Regression parameters: {grid_search.best_params_}")
    print(f"Best Logistic Regression score (CV Accuracy): {grid_search.best_score_}")
    
    best_model = grid_search.best_estimator_
    
    y_pred = best_model.predict(X_test_scaled)
    y_prob = best_model.predict_proba(X_test_scaled)[:,1]
    
    print("--- Tuned Logistic Regression Performance on Test Set ---")
    print(classification_report(y_test, y_pred))
    print(f"Test Set Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Test Set ROC AUC Score: {roc_auc_score(y_test, y_prob)}")
    
    # Plot Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=best_model.classes_)
    display.plot()
    plt.title("Tuned Logistic Regression Confusion Matrix (Scaled Features)")
    plt.show()
    
    # Plot coefficients
    coefs = best_model.coef_[0]
    coef_df = pd.DataFrame({'Feature': features_list, 'Coefficient': coefs})
    coef_df = coef_df.sort_values(by='Coefficient', ascending=False)
    
    plt.figure(figsize=(12, 7))
    sns.barplot(x='Coefficient', y='Feature', data=coef_df)
    plt.title("Tuned Logistic Regression Coefficients (Scaled Features)")
    plt.xlabel("Coefficient Value")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    RocCurveDisplay.from_estimator(best_model, X_test_scaled, y_test, name='Logistic Regression')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing (AUC = 0.5)')
    plt.title('Tuned Logistic Regression ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # just to get report to latex
    
    report = classification_report(y_test, y_pred, output_dict=True)
    df2 = pd.DataFrame(report).transpose()
    df2 = df2.round(4).reset_index()
    
    # Convert to LaTeX
    latex_table = df2.to_latex(index=False, escape=False)

    print(latex_table)
    
    return best_model

def run_pipeline():
    if os.path.exists(DATA_PROCESSED_PATH):
        df = pd.read_csv(DATA_PROCESSED_PATH)
        
    else:
        df = load_preprocess_raw(DATA_KAGGLE_RAW)
        df = get_yfinance_data(df)
        df = engineer_features(df)
        df.to_csv(DATA_PROCESSED_PATH, index=False)
    
    feature_list = ['EPS Estimate', 'Reported EPS', 'Surprise', 'Surprise scaled', 
                    'Surprise Direction', 'Surprise_scaled_x_Volatility', 'Market Return', 
                    'Volatility_x_Market_Return', 'Relative Surprise', 'Market_Return_sq',
                    'Momentum', 'Surprise_scaled_x_Market_Return', 'Volatility_sq', 
                    'Volatility_x_Momentum', 'Volatility', 'Surprise_scaled_sq',
                    'Momentum_sq', 'Market Volatility', 'Surprise_scaled_x_Momentum']
    
    perform_print_anova(df, feature_list, TARGET_COLUMN)
    
    # select model features
    X = df[SELECTED_FEATURES]
    y = df[TARGET_COLUMN]
    
    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE)
    
    # scale data
    scaler = StandardScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    best_model = train_evaluate_log_reg(X_train_scaled, X_test_scaled, y_train, y_test, SELECTED_FEATURES)
    
    # Save Model and Scaler
    print(f"Saving best model to {MODEL_SAVE_PATH} and scaler to {SCALER_SAVE_PATH}...")
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True) # Ensure models directory exists
    joblib.dump(best_model, MODEL_SAVE_PATH)
    joblib.dump(scaler, SCALER_SAVE_PATH)
    print("Model and scaler saved successfully.")

    print("\n--- Pipeline Execution Complete ---")
    
if __name__ == '__main__':
    run_pipeline()
    
    df = pd.read_csv(DATA_PROCESSED_PATH)
    
    feature_list = ['EPS Estimate', 'Reported EPS', 'Surprise', 'Surprise scaled', 
                    'Surprise Direction', 'Surprise_scaled_x_Volatility', 'Market Return', 
                    'Volatility_x_Market_Return', 'Relative Surprise', 'Market_Return_sq',
                    'Momentum', 'Surprise_scaled_x_Market_Return', 'Volatility_sq', 
                    'Volatility_x_Momentum', 'Volatility', 'Surprise_scaled_sq',
                    'Momentum_sq', 'Market Volatility', 'Surprise_scaled_x_Momentum']
    
    perform_print_anova(df, feature_list, TARGET_COLUMN)
    
    
    

















