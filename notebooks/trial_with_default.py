import os
import pandas as pd
from dotenv import load_dotenv

import evoml_client as ec
from evoml_client.trial_conf_models import BudgetMode

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


# Upload dataset to evoML
print("📤 Uploading dataset to evoML...")
data = pd.read_csv("../data/german_credit.csv")

trial_config = ec.TrialConfig.with_default(ec.MlTask.regression, BudgetMode.slow, ["Root Mean Squared Error"])

dataset = ec.Dataset.from_pandas(data, name="german_credit")
dataset.put()
dataset.wait()

trial, _ = ec.Trial.from_dataset_id(
    dataset.dataset_id,
    target_col="credit_amount",
    trial_name="german_credit",
    config=trial_config
)
trial.put()
trial.start()
trial.wait(900)

metrics_df = trial.get_metrics_dataframe()

# Get best model
best_model = trial.get_best()
best_model.build_model()

print()