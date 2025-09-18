import os
import pandas as pd
from dotenv import load_dotenv

import evoml_client as ec
from evoml_client.trial_conf_models import BudgetMode

from src.evoml_client.dataset import Dataset

# Load environment variables
load_dotenv()

# Configuration
API_URL = "https://evoml.ai"
EVOML_USERNAME = os.getenv("EVOML_USERNAME", "")
EVOML_PASSWORD = os.getenv("EVOML_PASSWORD", "")

# Initialize evoML client
try:
    ec.init(base_url=API_URL, username=EVOML_USERNAME, password=EVOML_PASSWORD)
    print("✅ Successfully connected to evoML platform")
except Exception as e:
    print(f"❌ Failed to connect to evoML: {e}")
    print("Please check your credentials in the .env file")

# data = pd.read_csv("../data/german_credit.csv")
# dataset = ec.Dataset.from_pandas(data, name="german_credit")
# dataset.put()
# dataset.wait()

dataset = Dataset.from_id("68ca9e2ae5732dd64210a63b")

task = ec.MlTask.regression
budget = BudgetMode.fast
loss_func = ["Root Mean Squared Error"]

# trial_config_with_default = ec.TrialConfig.with_default(task, budget, loss_func, is_timeseries=True)
# trial_with_default, _ = ec.Trial.from_dataset_id(
#     dataset.dataset_id,
#     target_col="credit_amount",
#     trial_name="german_credit",
#     config=trial_config_with_default
# )
# trial_with_default.put()
# trial_with_default.start()
# trial_with_default.wait(900)
# metrics_df_with_default = trial_with_default.get_metrics_dataframe()
# best_model_with_default = trial_with_default.get_best()
# best_model_with_default.build_model()

trial_config_with_models = ec.TrialConfig.with_models(
    models=["xgboost_regressor"], #["ridge_regressor", "lasso_regressor", "elastic_net_regressor"],
    task=task,
    budget_mode=budget,
    loss_funcs=loss_func,
    dataset_id=dataset.dataset_id,
    is_timeseries=True
)
trial_config_with_models.options.timeSeriesWindowSize = 6
trial_config_with_models.options.timeSeriesHorizon = 1
trial_config_with_models.options.enableBudgetTuning = False

df = dataset.data
column_index = df.columns.get_loc("time")
trial_config_with_models.options.timeSeriesColumnIndex = column_index
trial_name = "economic indicators"
target_column = "LCI_pct_change_lead_3"

trial_with_model, _ = ec.Trial.from_dataset_id(
    dataset.dataset_id,
    target_col=target_column,
    trial_name=trial_name,
    config=trial_config_with_models
)
trial_with_model.put()
trial_with_model.start()
trial_with_model.wait(900)
metrics_df_with_model = trial_with_model.get_metrics_dataframe()
best_model_with_model = trial_with_model.get_best()
best_model_with_model.build_model()

print()
