import evoml_client as ec
from evoml_client.trial_conf_models import BudgetMode, SplitMethodOptions
from dotenv import load_dotenv
import os

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

dataset = Dataset.from_id("68cbd5e2e5732dd6421155c2")

# config.options.timeSeriesWindowSize = 6
#             config.options.timeSeriesHorizon = horizon
#
# task = ec.MlTask.regression
#
# models = ec.get_allowed_models(task)
#
# loss_func = ec.get_allowed_loss_funcs(task)
#
# # result = get_data_info(dataset.dataset_id)
#
# ec.api_calls.get_permitted_models(task)["models"]

print()