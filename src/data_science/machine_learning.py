import logging
from typing import Union
from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd
from sklearn.pipeline import Pipeline


def create_error_logger() -> logging.Logger:
    """
    Creates a logger to record errors during pipeline execution.

    Returns:
        logging.Logger: The configured logger object.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.ERROR)
    return logger


def machine_learning(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    validate_set: pd.DataFrame,
    use_automl: bool,
    n_samples: int,
    time_limit: int,
    n_estimators: int,
    random_state: int,
    pipeline: Pipeline,
) -> Union[TabularPredictor, Pipeline]:
    """Trains a machine learning model using either AutoGluon or a provided pipeline.

    This function provides flexibility to train a model using either AutoGluon (automated machine learning)
    or a custom Scikit-learn pipeline. It handles potential errors related to data types, missing values,
    and unexpected issues during training.

    Args:
        x_train (pd.DataFrame): Features for the training set.
        y_train (pd.Series): Target variable for the training set.
        validate_set (pd.DataFrame): Features for the validation set.
        use_automl (bool): If True, use AutoGluon for training. If False, use the provided pipeline.
        n_samples (int, optional): Number of samples to use for AutoGluon training. If None, uses all samples.
        time_limit (int, optional): Time limit for AutoGluon training in seconds.
        n_estimators (int, optional): Number of estimators for Random Forest in the pipeline.
        random_state (int, optional): Random seed for reproducibility.
        pipeline (Pipeline, optional): A Scikit-learn pipeline to use when `use_automl` is False.

    Returns:
        Union[TabularPredictor, Pipeline]: The trained model (TabularPredictor if AutoGluon is used, otherwise the Pipeline).

    Raises:
        ValueError: If there are issues with data values or column names during training.
        KeyError: If a required column is missing from the dataset.
        TypeError: If `use_automl` is False and `pipeline` is None.
        Exception: For any other unexpected error during the training process.
    """
    logger = create_error_logger()
    try:
        if use_automl:
            # Merge with x_train and y_train
            train_data = x_train.sample(n=n_samples, random_state=random_state)
            train_data["Result"] = y_train.loc[train_data.index]

            # Debug: Print columns in train_data
            print("Columns in train_data for AutoML:", train_data.columns)
            predictor = TabularPredictor(
                label="Result", path="models", eval_metric="accuracy"
            ).fit(train_data, time_limit=time_limit, presets="medium_quality")
            return predictor
        else:
            if pipeline is None:
                logger.error("Pipeline is required when not using AutoML.")
                raise TypeError("Pipeline is required when not using AutoML.")

            # Adjust to training data
            pipeline.fit(x_train, y_train)
            x_train_processed = pipeline.named_steps["preprocessor"].transform(x_train)
            model = pipeline.named_steps["classifier"]
            model.fit(x_train_processed, y_train)
            return pipeline
    except ValueError as ve:
        logger.error("ValueError in machine learning process: %s", ve)
        raise
    except KeyError as ke:
        logger.error("KeyError in machine learning process: %s", ke)
        raise
    except Exception as e:
        logger.error("Unexpected error in machine learning process: %s", e)
        raise
