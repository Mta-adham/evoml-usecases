# Microsoft Stock Price Forecast

# Suppress warnings
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Setup & Dependencies
import getpass
from typing import Final
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
import plotly.graph_objects as go
from plotly.subplots import make_subplots
#import shap


import evoml_client as ec
from evoml_client.trial_conf_models import (
    BudgetMode,
    SplitMethodOptions,
)

API_URL: Final[str] = "https://evoml.ai"
EVOML_USERNAME: Final[str] = "phillipssasha02@gmail.com"
EVOML_PASSWORD: Final[str] = "RoundEffort29@"

# Connect to EvoML
ec.init(base_url=API_URL, username=EVOML_USERNAME, password=EVOML_PASSWORD)

print('check 1')


# Load & Basic Cleaning
stock = pd.read_csv("src/Microsoft_Stock.csv", index_col=0, parse_dates=True)
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
df["is_month_end"] = df.index.is_month_end.astype(int)

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

# Store dates separately for later use
df_dates = df.index.copy()

# Create dataset with ONLY target and features (no date column)
df_for_modeling = df[[target_col] + feature_cols].copy()

print('check 3')

# Split data properly
split_idx = int(len(df_for_modeling) * 0.8)
to_upload = df_for_modeling.iloc[:split_idx].copy()
holdout_df = df_for_modeling.iloc[split_idx:].copy()

print('check 4')

# Upload dataset (ONLY target and features, no date column)
dataset = ec.Dataset.from_pandas(to_upload, name="Microsoft_Stock_Forecast")
dataset.put()
dataset.wait()
print(f"Dataset URL: {API_URL}/platform/datasets/view/{dataset.dataset_id}")

print('check 5')

# Configures trial as Time Series (following working notebook pattern)
models = ["ridge_regressor", "lasso_regressor", "elastic_net_regressor"]

config = ec.TrialConfig.with_models(
    models=models,
    task=ec.MlTask.regression,
    budget_mode=BudgetMode.fast,
    loss_funcs=["Root Mean Squared Error"],
    dataset_id=dataset.dataset_id,
    is_timeseries=True,
)

# Time series specific configuration
config.options.timeSeriesWindowSize = 6
config.options.timeSeriesHorizon = 1  # Start with horizon 1 for testing
config.options.splittingMethodOptions = SplitMethodOptions(
    method="percentage", 
    trainPercentage=0.8
)
config.options.enableBudgetTuning = False

print('Check 6')

# starts trial
trial, dataset = ec.Trial.from_dataset_id(
    dataset.dataset_id,
    target_col=target_col,
    trial_name="MSFT Close Forecast",
    config=config,
)
trial.run(timeout=900)

# Check trial state before getting best model
print(f"Trial state: {trial.get_state()}")
if trial.get_state().name == "FAILED":
    print("Trial failed. Checking trial details...")
    # You can add more debugging here if needed
    raise Exception("Trial failed. Please check the trial configuration and data.")

# fetches the best performing model from the pipeline
best_model = trial.get_best()
best_model.build_model()

print('Check 7')

# Actual vs Predicted (validation data from training set)
val_len = max(1, int(len(to_upload) * 0.2))
validation_df = to_upload.iloc[-val_len:].copy()

# Get corresponding dates for validation set
validation_dates = df_dates[split_idx-val_len:split_idx]

print('Check 8')

# Make predictions
try:
    X_val = validation_df[feature_cols].copy()
    y_val = validation_df[target_col].copy()
    
    # Check for any NaN values in features
    if X_val.isnull().any().any():
        print("Warning: NaN values found in features. Dropping rows with NaN values.")
        X_val = X_val.dropna()
        y_val = y_val.loc[X_val.index]
        validation_dates = validation_dates[X_val.index]
    
    pred_val = best_model.predict(X_val)
    
    print(f"Prediction successful! Shape: {pred_val.shape}")
    print(f"Actual values shape: {y_val.shape}")
    print(f"Feature values shape: {X_val.shape}")
    
except Exception as e:
    print(f"Error during prediction: {e}")
    print(f"X_val shape: {X_val.shape}")
    print(f"X_val columns: {X_val.columns.tolist()}")
    print(f"X_val sample:\n{X_val.head()}")
    raise

print('Check 9')

# Plot against dates (shared x-axis)
fig = go.Figure()
fig.add_trace(go.Scatter(x=validation_dates, y=y_val, mode="lines", name="Actual"))
fig.add_trace(go.Scatter(x=validation_dates, y=pred_val, mode="lines", name="Predicted"))
fig.update_layout(
    title="MSFT Close: Actual vs Predicted (Holdout)",
    xaxis_title="Date",
    yaxis_title="Close (USD)",
    template="plotly_white",
    width=1000, height=400
)
fig.show()

# Additional validation: Check prediction quality
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse = mean_squared_error(y_val, pred_val)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_val, pred_val)
r2 = r2_score(y_val, pred_val)

print(f"\nPrediction Quality Metrics:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R² Score: {r2:.4f}")

# Plot residuals
residuals = y_val - pred_val
fig_residuals = go.Figure()
fig_residuals.add_trace(go.Scatter(x=validation_dates, y=residuals, mode="lines", name="Residuals"))
fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")
fig_residuals.update_layout(
    title="Prediction Residuals",
    xaxis_title="Date",
    yaxis_title="Residual (Actual - Predicted)",
    template="plotly_white",
    width=1000, height=400
)
fig_residuals.show()

