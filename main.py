import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import pandas as pd

# DATA SETUP

# Retrive the dataframe for the S&P 500
sp500 = yf.Ticker("^GSPC")
sp500 = sp500.history(period="max")

# Delete the 2 unnessesary columns
del sp500["Dividends"]
del sp500["Stock Splits"]

# Create a column called 'Tomorrow' where tomorrows price is the shift of 'Close' column by -1
sp500["Tomorrow"] = sp500["Close"].shift(-1)

# Create a column called 'Target' where it returns boolean converted to int (1 or 0), is tomorrows price greater than todays?
sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)

# Removes all data before 1990 using Pandas 'loc' method to remove discrepencies
sp500 = sp500.loc["1990-01-01":].copy()

# MODEL TRAINING

# Random Forest Classifier init
model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)

# With time series testing (like we are), we can't use cross validation

# Spliting data for testing and training, last 100 rows will be for testings, rest is training data
train = sp500.iloc[:-100]
test = sp500.iloc[-100:]

# Be specific when creating predictors to predict the target
predictors = ["Close", "Volume", "Open", "High", "Low"]

# Fit model
model.fit(train[predictors], train["Target"])

preds = model.predict(test[predictors]) # Test set, returns numPy array
preds = pd.Series(preds, index=test.index) # Convert to pd dataframe

# Prints the precision score (closer to 1 is better)
print(f"Initial Precision Score: {precision_score(test["Target"], preds)}")

"""
The RandomForestClassifier model has a precision score of 0.5, which is not good, we can make it better
"""

# Backtesting

# Function to train the model (everything we did in a function format)
def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1) # Takes the dataframes and combines into a single df
    return combined

# Backtesting function
def backtest(data, model, predictors, start=2500, step=250):
    all_predictions = []

    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    
    return pd.concat(all_predictions)

predictions = backtest(sp500, model, predictors)
print(f"Precision Score after Backtesting: {precision_score(predictions["Target"], predictions["Predictions"])}")

# Benchmark
benchmark = predictions["Target"].value_counts() / predictions.shape[0]
print(f"Benchmark: {benchmark}")

# Rolling averages
horizons = [2, 5, 60, 250, 1000]
new_predictors = []

for horizon in horizons:
    rolling_averages = sp500.rolling(horizon).mean()

    ratio_column = f"Close_Ratio_{horizon}"
    sp500[ratio_column] = sp500["Close"] / rolling_averages["Close"]

    trend_column = f"Trend_{horizon}"
    sp500[trend_column] = sp500.shift(1).rolling(horizon).sum()["Target"]

    new_predictors += [ratio_column, trend_column]

sp500 = sp500.dropna() # Drop rows in which the data shows 'NaN'

# UPDATING MODEL

model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)

def updated_predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:,1] # Returns probability
    preds[preds >= 0.75] = 1
    preds[preds < 0.75] = 0
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1) # Takes the dataframes and combines into a single df
    return combined

predictions = backtest(sp500, model, new_predictors)

new_prediction_score = precision_score(predictions["Target"], predictions["Predictions"])
print(f"Updated Precision Score: {new_prediction_score}")