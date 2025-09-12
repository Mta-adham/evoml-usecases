# Microsoft Stock Price Forecast

# Setup & Dependencies
import getpass
from typing import Final
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import plotly.graph_objects as go
from plotly.subplots import make_subplots
#import shap

import evoml_client as ec
from evoml_client.trial_conf_models import (
    BudgetMode,
    HoldoutOptions,
    ValidationMethod,
    ValidationMethodOptions,
)

API_URL: Final[str] = "https://evoml.ai"
EVOML_USERNAME: Final[str] = "phillipssasha02@gmail.com"
EVOML_PASSWORD: Final[str] = "RoundEffort29@"

# Connect to EvoML
ec.init(base_url=API_URL, username=EVOML_USERNAME, password=EVOML_PASSWORD)

print('check 1')






# Load & Basic Cleaning
stock = pd.read_csv("Microsoft_Stock.csv", index_col=0, parse_dates=True)
stock.columns = stock.columns.str.lower().str.replace("-", "_")
# Ensure expected columns exist
required = {"open", "high", "low", "close", "volume"}
missing = required - set(stock.columns)
if missing:
    raise ValueError(f"Your CSV is missing expected columns: {missing}")


fig = go.Figure()
fig.add_trace(go.Scatter(x=stock.index, y=stock["close"], mode="lines", name="Close"))
fig.update_layout(
    title="Microsoft Daily Close Price",
    xaxis_title="Date",
    yaxis_title="Close (USD)",
    template="plotly_white",
    width=1000, height=400
)
fig.show()

hist_fig = go.Figure()
hist_fig.add_trace(go.Histogram(x=stock["close"], nbinsx=50))
hist_fig.update_layout(
    title="Distribution of Close Prices",
    xaxis_title="Close (USD)",
    yaxis_title="Frequency",
    template="plotly_white",
    width=1000, height=400
)
hist_fig.show()






print('check 2')

# Feature Engineering
df = stock.copy()
df["is_monday"] = (df.index.weekday == 0).astype(int)
df["is_month_start"] = df.index.is_month_start.astype(int)
df["is_month_end"]   = df.index.is_month_end.astype(int)

df["close_lag_1"] = df["close"].shift(1)
df["close_lag_5"] = df["close"].shift(5)
df["close_lag_10"] = df["close"].shift(10)

df["close_roll_mean_5"] = df["close"].rolling(5).mean()
df["close_roll_mean_10"] = df["close"].rolling(10).mean()
df["close_roll_std_5"] = df["close"].rolling(5).std()
df["close_roll_std_10"] = df["close"].rolling(10).std()

df["return_1"] = df["close"].pct_change()
df = df.dropna().copy()

target_col = "close"
feature_cols = [
    "is_monday", "is_month_start", "is_month_end",
    "close_lag_1", "close_lag_5", "close_lag_10",
    "close_roll_mean_5", "close_roll_mean_10",
    "close_roll_std_5", "close_roll_std_10",
    "return_1",
]

df = df[[target_col] + feature_cols].copy()

split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx].copy()
holdout_df = df.iloc[split_idx:].copy()

target_col = "close"

print('check 3')


# Upload dataset (with date column)
to_upload = df.reset_index().rename(columns={"index": "date"})

dataset = ec.Dataset.from_pandas(to_upload, name="Microsoft_Stock_Forecast")
dataset.put()
dataset.wait()
print(f"Dataset URL: {API_URL}/platform/datasets/view/{dataset.dataset_id}")

print('check 4')

# Configures trial as Time Series
config = ec.TrialConfig.with_default(
    task=ec.MlTask.regression,
    budget_mode=BudgetMode.fast,
    loss_funcs=["Root Mean Squared Error"],
    dataset_id=dataset.dataset_id,
    is_timeseries=True
)
config.options.timeSeriesHorizon = 10 # sets horizon to 10 (this number can be changed to between 1 and 100)
config.options.enableBudgetTuning = False
config.options.validationMethodOptions = ValidationMethodOptions(
    method=ValidationMethod.holdout,
    holdoutOptions=HoldoutOptions(size=0.2, keepOrder=True)
)

print('Check 5')




# starts trial
trial, dataset = ec.Trial.from_dataset_id(
    dataset.dataset_id,
    target_col=target_col,
    trial_name="MSFT Close Forecast",
    config=config,
)
trial.run(timeout=900)

# fetches the best performing model from the pipeline
best_model = trial.get_best()
best_model.build_model()

print('Check 6')



# Actual vs Predicted (test data)
val_len = max(1, int(len(to_upload) * 0.2))
if "date" not in to_upload.columns:
    if to_upload.index.name:
        to_upload = to_upload.reset_index()
        if "date" not in to_upload.columns:
            first_col = to_upload.columns[0]
            to_upload = to_upload.rename(columns={first_col: "date"})
    else:
        to_upload = to_upload.reset_index().rename(columns={to_upload.columns[0]: "date"})

validation_df = to_upload.iloc[-val_len:].copy()
if "date" not in validation_df.columns:
    raise KeyError("Could not find a 'date' column in the uploaded DataFrame; "
                   "expected to_upload to contain a 'date' column or a datetime index.")

validation_df["date"] = pd.to_datetime(validation_df["date"], errors="coerce")


print('Check 7')


X_val = validation_df[feature_cols].copy()
y_val = validation_df[target_col].copy()
pred_val = best_model.predict(X_val)





print('Check 8')

# Plot against dates (shared x-axis)
fig = go.Figure()
fig.add_trace(go.Scatter(x=validation_df["date"], y=y_val, mode="lines", name="Actual"))
fig.add_trace(go.Scatter(x=validation_df["date"], y=pred_val, mode="lines", name="Predicted"))
fig.update_layout(
    title="MSFT Close: Actual vs Predicted (Holdout)",
    xaxis_title="Date",
    yaxis_title="Close (USD)",
    template="plotly_white",
    width=1000, height=400
)
fig.show()