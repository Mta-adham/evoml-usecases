from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt
from typing import Final
import os

import evoml_client as ec
from evoml_client.trial_conf_models import BudgetMode

load_dotenv()

# Load credentials from environment variables
API_URL: Final[str] = "https://evoml.ai"
EVOML_USERNAME: Final[str] = os.environ.get("EVOML_USERNAME")
EVOML_PASSWORD: Final[str] = os.environ.get("EVOML_PASSWORD")

# Connect to the evoML platform
ec.init(base_url=API_URL, username=EVOML_USERNAME, password=EVOML_PASSWORD)

cumulative_energy = pd.read_csv("../data/all-users-daily-data.csv")
cumulative_energy["energy"] *= 1e-10  # Convert from raw meter readings to kWh

# Function to standardize time index and compute daily energy
def calculate_daily_energy(df):
    return (
        df.assign(date=pd.to_datetime(df['date'])) # Ensure date column is in datetime format
        .set_index('date')                         # Set date column as the index
        .resample('D')                             # Resample to ensure daily frequency
        .asfreq()                                  # Fill missing dates with NaNs
        .assign(daily_energy=lambda x: x['energy'].diff())  # Compute daily energy difference
    )

# Apply the function groupwise and standardize the time index
daily_energy = cumulative_energy.groupby('userId').apply(calculate_daily_energy)

# Group by date and calculate the mean daily_energy
mean_daily_energy = daily_energy.pivot_table(values='daily_energy', index='date', aggfunc='mean')

mean_daily_energy["daily_energy"].plot(figsize=(15, 6), linestyle='-')
plt.title('Mean Daily Energy Consumption Across All Users Over Time')
plt.xlabel('Date')
plt.ylabel('Mean Daily Energy Consumption')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# Upload the dataset
dataset = ec.Dataset.from_pandas(mean_daily_energy.reset_index(), name="Mean Daily Energy Consumption")
dataset.put()
dataset.wait()
print(f"Dataset URL: {API_URL}/platform/datasets/view/{dataset.dataset_id}")

config = ec.TrialConfig.with_default(
    task=ec.MlTask.regression,
    budget_mode=BudgetMode.fast,
    loss_funcs=["R2"],
    dataset_id=dataset.dataset_id,
    is_timeseries=True,
)
config.options.timeSeriesWindowSize = 7
config.options.enableBudgetTuning = False

trial, dataset = ec.Trial.from_dataset_id(
    dataset.dataset_id,
    target_col="daily_energy",
    trial_name="Forecast - Daily Energy",
    config=config,
)

trial.run(timeout=900)

metrics = trial.get_metrics_dataframe()
selected_metrics = metrics.loc[:, pd.IndexSlice["regression-r2", ["validation", "test"]]]
selected_metrics.sort_values(by=("regression-r2", "validation"), ascending=False)

# Retrieve the best model according to the specified objective function and validation set
best_model = trial.get_best()

# Build the model so that we can use it for prediction
best_model.build_model()

# Select n_days of data
n_days = 7*4
test_data = mean_daily_energy.iloc[-n_days:]

# Predict the mean daily energy
predicted_mean_daily_energy = pd.Series(best_model.predict(data=test_data.reset_index()), index=test_data.index)

# Plot the actual and predicted mean daily energy without auto-legend
ax = test_data.plot(
    figsize=(15, 6),
    linestyle='-',
    label='Actual',
    legend=False
)
predicted_mean_daily_energy.plot(
    ax=ax,
    linestyle='--',
    label='Predicted',
    legend=False
)
ax.legend(['Actual', 'Predicted'])

plt.title('Mean Daily Energy Consumption: Actual vs Predicted')
plt.xlabel('Date')
plt.ylabel('Mean Daily Energy Consumption')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()