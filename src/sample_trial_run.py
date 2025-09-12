from pathlib import Path
from typing import List, Tuple

import pandas as pd
from evoml_client import MlTask, Model, Trial, TrialConfig
from evoml_client.api_calls import get_trial_pipelines, trial_get
from loguru import logger

from src.client.client import initialise_client
from src.configs.create_config import get_default_config
from src.configs.test_auto_config import TestTrialConfig
from src.trial_options.loss_function import get_default_loss_functions
from src.trial_options.model import get_random_models
from src.utils.dataset import get_task


def load_dataset(file_path: Path) -> pd.DataFrame:
    """Load the dataset from a CSV file."""
    logger.info(f"Loading dataset from {file_path}")
    return pd.read_csv(str(file_path))


def get_trial_configuration(
    trial_name: str,
    df: pd.DataFrame,
    task: MlTask,
    target: str,
    tags: List[str],
    is_timeseries=False,
) -> Tuple[TrialConfig, TestTrialConfig]:
    """Get the trial configuration based on the task."""
    selected_loss_functions = get_default_loss_functions(task)
    selected_models = get_random_models(task, 1)
    default_config = get_default_config(task, selected_models, selected_loss_functions)
    label_index = df.columns.get_loc(target)

    return default_config, TestTrialConfig(
        trial_name=trial_name,
        dataset_name=target,
        target=target,
        target_index=label_index,
        tags=tags,
        is_timeseries=is_timeseries,
    )


def run_trial(df: pd.DataFrame, default_config: dict, config: TestTrialConfig) -> Trial:
    """Run the trial and return the trial reference."""
    trial_ref, dataset_ref = Trial.from_pandas(
        data=df,
        data_name=config.dataset_name,
        target_col=config.target_index,
        trial_name=config.trial_name,
        config=default_config,
        tags=config.tags,
    )

    logger.info(f"Running trial: {config.trial_name}")
    trial_ref.run(timeout=900)
    logger.info(f"Trial complete status: {trial_ref.get_state()}")

    return trial_ref


def get_best_pipeline(trial_ref: Trial):
    trial_id = trial_ref.trial_id
    all_pipelines = get_trial_pipelines(trial_id)
    return trial_ref.get_best_pipeline(all_pipelines, None, None)


def get_best_model(trial_ref: Trial) -> Model:
    """Get the best model from the trial."""
    best = trial_ref.get_best()
    best.build_model()
    return best


def make_predictions(model: Model, df: pd.DataFrame) -> pd.DataFrame:
    """Make predictions using the model."""
    predictions = model.predict(df)
    print(predictions)
    return predictions


def main():
    # authenticate evoml client
    initialise_client()

    # specify dataset and trial names
    file_name = "amazon"
    label_name = "target"
    tags = ["sample"]

    # load dataset and identify task
    base_dir = Path(__file__).resolve().parent.parent.parent
    full_path = base_dir / f"datasets/standard/classification_binary/{file_name}.csv"
    df = pd.read_csv(str(full_path))

    # df = do_feature_engineering(df)

    task = get_task(df, label_name)
    is_timeseries = False

    # Get trial configuration
    default_config, config = get_trial_configuration(
        file_name, df, task, label_name, tags, is_timeseries
    )

    # Run the trial
    trial_ref = run_trial(df, default_config, config)

    # Get trial files
    trial_id = trial_ref.trial_id

    # Get the best model and make predictions
    model = get_best_model(trial_id)

    # Run prediction
    result = make_predictions(model, df)

    logger.info(f"Trial complete status: {trial_ref.get_state()}")


if __name__ == "__main__":
    main()
