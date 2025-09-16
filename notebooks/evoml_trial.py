from dotenv import load_dotenv
import evoml_client as ec
from evoml_client.trial_conf_models import (
    BudgetMode,
    HoldoutOptions,
    SplitMethodOptions,
    ValidationMethod,
    ValidationMethodOptions,
)
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import plotly.subplots as sp
import pycountry
from typing import Final

load_dotenv()

API_URL: Final[str] = "https://evoml.ai"
EVOML_USERNAME: Final[str] = os.getenv("EVOML_USERNAME")
EVOML_PASSWORD: Final[str] = os.getenv("EVOML_PASSWORD")

ec.init(base_url=API_URL, username=EVOML_USERNAME, password=EVOML_PASSWORD)

# Load dataset
df = pd.read_csv("./data/processed/economic_indicators_quarterly_yoy.csv")
df["time"] = pd.to_datetime(df["time"])

# generate lags on dataset:
print(df.columns)

target = "LCI_pct_change"
lead_num = 3  # Define how many periods ahead to forecast (1 = next quarter, 2 = two quarters ahead, etc.)


def generate_lags(df, max_lags):
    """
    Generate multiple lag features for numeric columns in the dataframe

    Args:
        df: pandas DataFrame
        max_lags: maximum number of lags to generate

    Returns:
        DataFrame with lag features added
    """
    df_copy = df.copy()

    for col in df.columns:
        # Skip lag generation for country, time, and target column
        if col not in ["country", "time", target]:
            for lag in range(1, max_lags + 1):
                # Group by country to ensure lags are calculated within each country
                df_copy[f"{col}_lag_{lag}"] = df_copy.groupby("country")[col].shift(lag)

    return df_copy


# Generate lags up to 4 periods
df = generate_lags(df, max_lags=4)

print(df.head())


def generate_lead_target(df, lead_periods):
    """
    Generate a lead target column for the dataframe

    Args:
        df: pandas DataFrame
        lead_periods: number of periods ahead to forecast

    Returns:
        DataFrame with lead target column added
    """
    df_copy = df.copy()
    # Group by country to ensure lead is calculated within each country
    df_copy[f"{target}_lead_{lead_periods}"] = df_copy.groupby("country")[target].shift(
        -lead_periods
    )
    return df_copy


df = generate_lead_target(df, lead_periods=lead_num)

lead_target = f"{target}_lead_{lead_num}"

# drop original target column to prevent data leakage.
df = df.drop(columns=[target])

# --- Check if lags exist within the target variable ---
lead_lag_cols = [col for col in df.columns if col.startswith(f"{lead_target}_lag")]
if lead_lag_cols:
    df = df.drop(columns=lead_lag_cols)
    print(f"Dropped lead lag columns: {lead_lag_cols}")


# Sort by time in order to ensure continuous time series and a representations of countries.
df = df.sort_values("time").reset_index(drop=True)
print(f"\nData sorted by time. Date range: {df['time'].min()} to {df['time'].max()}")
print(f"First few dates: {df['time'].head(10).tolist()}")
print(f"Last few dates: {df['time'].tail(10).tolist()}")

# remove rows with NaN values
df = df.dropna()

# Upload dataset into evoml:
dataset = ec.Dataset.from_pandas(df, name="Economic Indicators")
dataset.put()
dataset.wait()

print(f"Dataset URL: {API_URL}/platform/datasets/view/{dataset.dataset_id}")

# Configure the trial:

config = ec.TrialConfig.with_models(
    models=[
        "ridge_regressor",
        "bayesian_ridge_regressor",
        "linear_regressor",
        "lasso_regressor",
    ],
    task=ec.MlTask.regression,
    budget_mode=BudgetMode.fast,
    loss_funcs=["Root Mean Squared Error"],
    dataset_id=dataset.dataset_id,
)

config.options.enableBudgetTuning = False
config.options.validationMethodOptions = ValidationMethodOptions(
    method=ValidationMethod.holdout,
    holdoutOptions=HoldoutOptions(size=0.2, keepOrder=True),
)

trial, _ = ec.Trial.from_dataset_id(
    dataset.dataset_id,
    target_col=lead_target,  # Use the lead target for forecasting
    trial_name=f"Labour_cost_forecast_{lead_num}_period_ahead",
    config=config,
)

trial.run(timeout=900)

best_model = trial.get_best()
# Build the model
best_model.build_model()


def generate_predicted_actual_visualisation(test_data):
    # Get predictions
    predictions = best_model.predict(test_data)

    # Create a copy of test data and add predictions
    plot_data = test_data.copy()
    plot_data["Predicted"] = predictions
    plot_data["Actual"] = plot_data[f"{lead_target}"]

    # Create output directory if it doesn't exist
    os.makedirs(f"data/Model_output_{lead_num}Q", exist_ok=True)

    # Extract year from time column for aggregation
    plot_data["year"] = pd.to_datetime(plot_data["time"]).dt.year

    # First, calculate annual averages for each country (bottom-up aggregation from quarterly data)
    country_annual_data = (
        plot_data.groupby(["country", "year"])
        .agg({"Actual": "mean", "Predicted": "mean"})
        .reset_index()
    )

    # Then, calculate the average across all countries for each year
    overall_annual_data = (
        country_annual_data.groupby("year")
        .agg({"Actual": "mean", "Predicted": "mean"})
        .reset_index()
    )

    # Remove any rows with NaN values
    overall_annual_data = overall_annual_data.dropna()

    # Create figure for overall annual comparison
    fig = go.Figure()

    # Add actual annual averages
    fig.add_trace(
        go.Bar(
            x=overall_annual_data["year"],
            y=overall_annual_data["Actual"],
            name="Actual Annual Average",
            marker_color="lightblue",
            opacity=0.8,
        )
    )

    # Add predicted annual averages
    fig.add_trace(
        go.Bar(
            x=overall_annual_data["year"],
            y=overall_annual_data["Predicted"],
            name="Predicted Annual Average",
            marker_color="orange",
            opacity=0.8,
        )
    )

    # Update layout for better visualization
    fig.update_layout(
        title=f"Overall Annual LCI Forecasting - All Countries Average<br>Average Year-over-Year % Change",
        xaxis_title="Year",
        yaxis_title="Average YoY % Change in LCI",
        legend_title="Type",
        barmode="group",  # Group bars side by side
        height=600,
        width=1000,
        showlegend=True,
        template="plotly_white",
    )

    # Update x-axis to show years properly
    fig.update_xaxes(tickmode="linear", dtick=1, tickformat="d")

    # Add hover information
    fig.update_traces(
        hovertemplate="<b>%{fullData.name}</b><br>"
        + "Year: %{x}<br>"
        + "Average YoY Change: %{y:.2f}%<br>"
        + "<extra></extra>"
    )

    # Save plot to HTML file
    output_path = f"data/Model_output_{lead_num}Q/annual_forecast_overall.html"
    fig.write_html(output_path)

    # Display the plot
    fig.show()

    # Also try to open in browser as backup
    try:
        import webbrowser

        webbrowser.open(output_path)
        print(f"Opened visualization in browser: {output_path}")
    except Exception as e:
        print(f"Could not open browser: {e}")
        print(f"You can manually open the HTML file at: {output_path}")

    return fig

# def generate_country_specific_visualisations(test_data):
#     """
#     Generate individual HTML plots for each country with predicted vs actual values
# 
#     Args:
#         test_data: pandas DataFrame containing test data
# 
#     Returns:
#         dict: Dictionary containing figures for each country
#     """
#     # Get predictions
#     predictions = best_model.predict(test_data)
# 
#     # Create a copy of test data and add predictions
#     plot_data = test_data.copy()
#     plot_data["Predicted"] = predictions
#     plot_data["Actual"] = plot_data[f"{lead_target}"]
# 
#     # Create output directory if it doesn't exist
#     os.makedirs(f"data/Model_output_{lead_num}Q", exist_ok=True)
# 
#     # Extract year from time column for aggregation
#     plot_data["year"] = pd.to_datetime(plot_data["time"]).dt.year
# 
#     # Get unique countries
#     countries = plot_data["country"].unique()
#     country_figures = {}
# 
#     for country in countries:
#         # Filter data for specific country
#         country_data = plot_data[plot_data["country"] == country].copy()
# 
#         # Calculate annual averages for this country
#         country_annual_data = (
#             country_data.groupby("year")
#             .agg({"Actual": "mean", "Predicted": "mean"})
#             .reset_index()
#         )
# 
#         # Remove any rows with NaN values
#         country_annual_data = country_annual_data.dropna()
# 
#         if len(country_annual_data) == 0:
#             print(f"Warning: No data available for {country}")
#             continue
# 
#         # Create figure for this country
#         fig = go.Figure()
# 
#         # Add actual annual averages
#         fig.add_trace(
#             go.Bar(
#                 x=country_annual_data["year"],
#                 y=country_annual_data["Actual"],
#                 name="Actual Annual Average",
#                 marker_color="lightblue",
#                 opacity=0.8,
#             )
#         )
# 
#         # Add predicted annual averages
#         fig.add_trace(
#             go.Bar(
#                 x=country_annual_data["year"],
#                 y=country_annual_data["Predicted"],
#                 name="Predicted Annual Average",
#                 marker_color="orange",
#                 opacity=0.8,
#             )
#         )
# 
#         # Update layout for better visualization
#         fig.update_layout(
#             title=f"LCI Forecasting - {country}<br>Annual Year-over-Year % Change",
#             xaxis_title="Year",
#             yaxis_title="YoY % Change in LCI",
#             legend_title="Type",
#             barmode="group",  # Group bars side by side
#             height=600,
#             width=1000,
#             showlegend=True,
#             template="plotly_white",
#         )
# 
#         # Update x-axis to show years properly
#         fig.update_xaxes(tickmode="linear", dtick=1, tickformat="d")
# 
#         # Add hover information
#         fig.update_traces(
#             hovertemplate="<b>%{fullData.name}</b><br>"
#             + "Year: %{x}<br>"
#             + "YoY Change: %{y:.2f}%<br>"
#             + "<extra></extra>"
#         )
# 
#         # Save plot to HTML file
#         country_safe_name = country.replace(" ", "_").replace("/", "_")
#         output_path = (
#             f"data/Model_output_{lead_num}Q/annual_forecast_{country_safe_name}.html"
#         )
#         fig.write_html(output_path)
#         print(f"Saved {country} annual forecast visualization to {output_path}")
# 
# 
# # Get test data from the model and generate visualizations
# test_data = df.iloc[int(-len(df) * 0.2) :]
# print("Generating visualization...")
# fig = generate_predicted_actual_visualisation(test_data)
# print("Visualization generation complete!")
# 
# # Generate country-specific visualizations
# print("\nGenerating country-specific visualizations...")
# country_figs = generate_country_specific_visualisations(test_data)
# print("Country-specific visualization generation complete!")
