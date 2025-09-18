import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import evoml_client as ec
from evoml_client.trial_conf_models import BudgetMode, SplitMethodOptions
import os
from dotenv import load_dotenv
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

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


class LCIDataProcessor:
    """Handles LCI data loading and preprocessing - following CPI pattern exactly"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.raw_data = None
        self.processed_data = None
        
    def load_data(self) -> pd.DataFrame:
        """Load LCI data from CSV file"""
        try:
            # Load the data
            self.raw_data = pd.read_csv(self.data_path)
            
            # Convert time column to datetime
            self.raw_data['time'] = pd.to_datetime(self.raw_data['time'])
            
            print(f"✅ Loaded {len(self.raw_data)} rows of LCI data")
            print(f"📅 Date range: {self.raw_data['time'].min()} to {self.raw_data['time'].max()}")
            print(f"🌍 Countries: {self.raw_data['country'].nunique()}")
            
            return self.raw_data
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def generate_lags(self, df: pd.DataFrame, max_lags: int = 4) -> pd.DataFrame:
        """Generate lag features for time series analysis"""
        df_copy = df.copy()
        
        for col in df.columns:
            # Skip lag generation for country, time, and target column
            if col not in ["country", "time", "LCI_pct_change"]:
                for lag in range(1, max_lags + 1):
                    # Group by country to ensure lags are calculated within each country
                    df_copy[f"{col}_lag_{lag}"] = df_copy.groupby("country")[col].shift(lag)
        
        print(f"✅ Generated lag features up to {max_lags} periods")
        return df_copy
    
    def process_data(self) -> pd.DataFrame:
        """Complete data processing pipeline - following CPI pattern exactly"""
        # Load data
        df = self.load_data()
        if df is None:
            return None
        
        # Generate lags
        df = self.generate_lags(df, max_lags=4)
        
        # Sort by time to ensure continuous time series
        df = df.sort_values("time").reset_index(drop=True)
        
        # Ensure time column is properly formatted as datetime
        df['time'] = pd.to_datetime(df['time'])
        
        # Remove rows with NaN values
        initial_rows = len(df)
        df = df.dropna()
        final_rows = len(df)
        
        print(f"✅ Data processing complete: {initial_rows} → {final_rows} rows")
        print(f"📅 Time column type: {df['time'].dtype}")
        print(f"📅 Time range: {df['time'].min()} to {df['time'].max()}")
        
        self.processed_data = df
        return df
    
    def get_analysis_data(self) -> pd.DataFrame:
        """Get processed data for analysis - following CPI pattern exactly"""
        if self.processed_data is None:
            print("❌ No processed data available")
            return None
        # Return data with time and LCI_pct_change columns only, like CPI pattern
        return self.processed_data[['time', 'LCI_pct_change']].copy()
    
    def get_visualization_data(self) -> pd.DataFrame:
        """Get data for visualization"""
        if self.processed_data is None:
            return None
        return self.processed_data[['time', 'LCI_pct_change']].copy()

# Simplified LCIForecaster following CPI pattern exactly
class LCIForecaster:
    """Handles LCI forecasting with multiple horizons - following CPI pattern exactly"""
    
    def __init__(self, dataset_id: str):
        self.dataset_id = dataset_id
        self.trials = {}
        self.results = {}
        
    def create_trial(self, horizon: int, trial_name: str) -> Optional[object]:
        """Create and run a trial for a specific horizon - following CPI pattern exactly"""
        try:
            print(f"🚀 Creating trial for {horizon}-quarter horizon...")
            
            # Configure trial exactly like CPI notebook
            config = ec.TrialConfig.with_models(
                models=["xgboost_regressor"], #["ridge_regressor", "lasso_regressor", "elastic_net_regressor"],
                task=ec.MlTask.regression,
                budget_mode=BudgetMode.fast,
                loss_funcs=["Root Mean Squared Error"],
                dataset_id=self.dataset_id,
                is_timeseries=True
            )
            
            # Set time series parameters exactly like CPI notebook
            config.options.timeSeriesWindowSize = 6
            config.options.timeSeriesHorizon = horizon
            config.options.splittingMethodOptions = SplitMethodOptions(
                method="percentage", 
                trainPercentage=0.8
            )
            config.options.enableBudgetTuning = False
            
            # Create and run trial exactly like CPI notebook
            # Use the base target column, evoML will generate lead targets automatically
            trial, _ = ec.Trial.from_dataset_id(
                self.dataset_id,
                target_col="LCI_pct_change",
                trial_name=trial_name,
                config=config
            )
            
            trial.run(timeout=900)
            
            # Store trial and extract results exactly like CPI notebook
            self.trials[horizon] = trial
            self._extract_trial_results(trial, horizon)
            
            print(f"✅ Trial for {horizon}-quarter horizon completed successfully")
            return trial
            
        except Exception as e:
            print(f"❌ Error creating trial for {horizon}-quarter horizon: {e}")
            return None
    
    def _extract_trial_results(self, trial: object, horizon: int):
        """Extract results from a completed trial - following CPI pattern exactly"""
        try:
            # Get metrics
            metrics_df = trial.get_metrics_dataframe()
            
            # Get best model
            best_model = trial.get_best()
            best_model.build_model()
            
            # Extract model info
            model_rep_dict = best_model.model_rep.__dict__
            best_model_name = model_rep_dict.get('name')
            best_model_mse = model_rep_dict.get('metrics', {}).get('regression-mse', {}).get('test', {}).get('average')
            best_model_rmse = np.sqrt(best_model_mse) if best_model_mse else None
            
            # Store results
            self.results[horizon] = {
                'trial': trial,
                'best_model': best_model,
                'model_name': best_model_name,
                'mse': best_model_mse,
                'rmse': best_model_rmse,
                'metrics_df': metrics_df
            }
            
            print(f"📊 Best model for {horizon}-quarter: {best_model_name}")
            print(f"📈 RMSE: {best_model_rmse:.4f}")
            
        except Exception as e:
            print(f"❌ Error extracting results for {horizon}-quarter horizon: {e}")
    
    def run_all_trials(self, horizons: List[int] = [1, 3, 6, 9]):
        """Run trials for all specified horizons - following CPI pattern exactly"""
        print(f"🎯 Running trials for horizons: {horizons}")
        
        for horizon in horizons:
            trial_name = f"LCI_Forecast_{horizon}Q"
            self.create_trial(horizon, trial_name)
            print(f"\n{'='*50}\n")
        
        print(f"✅ Completed all trials. Results available for: {list(self.results.keys())}")
    
    def get_prediction_summary(self) -> pd.DataFrame:
        """Get summary of all predictions"""
        if not self.results:
            print("❌ No results available. Run trials first.")
            return None
        
        summary_data = []
        for horizon, result in self.results.items():
            summary_data.append({
                'Horizon (quarters)': horizon,
                'Best Model': result['model_name'],
                'RMSE': result['rmse'],
                'MSE': result['mse']
            })
        
        return pd.DataFrame(summary_data)
        
# Initialize data processor
processor = LCIDataProcessor("/home/manal/Workspace/evoml-usecases/data/processed/economic_indicators_quarterly_yoy.csv")

# Process data - following CPI pattern exactly
analysis_data = processor.process_data()

if analysis_data is not None:
    print("\n📊 Data Summary:")
    print(f"Shape: {analysis_data.shape}")
    print(f"Columns: {list(analysis_data.columns)}")
    print(f"\nFirst few rows:")
    print(analysis_data.head())
else:
    print("❌ Failed to process data")

# Upload dataset to evoML
print("📤 Uploading dataset to evoML...")
analysis_data = processor.get_analysis_data()

if analysis_data is not None:
    dataset = ec.Dataset.from_pandas(analysis_data, name="LCI_Dataset_Simplified")
    dataset.put()
    dataset.wait()
    print(f"✅ Dataset uploaded successfully. ID: {dataset.dataset_id}")
    
    # Initialize forecaster
    forecaster = LCIForecaster(dataset.dataset_id)
    
    # Run trials for all horizons
    forecaster.run_all_trials(horizons=[1])
    
    # Display results summary
    summary = forecaster.get_prediction_summary()
    if summary is not None and len(summary) > 0:
        print("\n📊 Prediction Results Summary:")
        print(summary.to_string(index=False))
    else:
        print("\n⚠️  No results available")
else:
    print("❌ No analysis data available")
