import evoml_client as ec

import os
import warnings
warnings.filterwarnings('ignore')
from dotenv import load_dotenv
from typing import Dict, List, Tuple, Optional

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

trial_id = "68caa500e5732dd64210baa9"
trial, _ = ec.Trial.from_id(trial_id)

metrics_df = trial.get_metrics_dataframe()

best_model = trial.get_best()
best_model.build_model()

