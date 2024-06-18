import logging
from typing import Union
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
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


def evaluate_model(
    x_test: pd.DataFrame,
    y_test: pd.Series,
    predictor: Union[TabularPredictor, Pipeline],
    n_samples_evaluate: int,
    random_state_evaluate: int,
):
    """Evaluates a trained machine learning model's performance on a test dataset.

    This function handles evaluation for two types of models:
        - AutoGluon's TabularPredictor
        - Scikit-learn Pipeline with a preprocessor and classifier steps

    If the predictor is a TabularPredictor, it evaluates using the built-in leaderboard and prints predictions.
    If the predictor is a Scikit-learn Pipeline, it first preprocesses a sample of the test data, makes predictions,
    and then calculates common classification metrics (accuracy, precision, recall, F1-score) and the confusion matrix.
    The results are printed to the console.

    Args:
        x_test (pd.DataFrame): Features for the test set.
        y_test (pd.Series): True labels for the test set.
        predictor: The trained machine learning model. Must be either:
            - TabularPredictor: An AutoGluon predictor object.
            - Pipeline: A Scikit-learn pipeline that includes a preprocessor step (named "preprocessor")
                        and a classifier step (named "classifier").
        n_samples_evaluate (int): Number of samples to randomly select from the test set for evaluation.
        random_state_evaluate (int): Random seed for reproducibility of sampling.

    Raises:
        ValueError:
            - If the input data for prediction is invalid or incorrectly formatted.
            - If metric calculation fails due to mismatched input or unsupported types.
        KeyError: If a required column or step name is missing in the input data or pipeline.
    """
    logger = create_error_logger()
    try:
        if isinstance(predictor, TabularPredictor):
            # AutoGluon TabularPredictor
            test_data = TabularDataset(
                x_test.sample(
                    n=n_samples_evaluate,
                    random_state=random_state_evaluate,
                    replace=True,
                )
            )
            predictions = predictor.predict(test_data)
            print(predictor.leaderboard())
            print(predictions)
        else:
            # Scikit-learn Pipeline
            # Sample and preprocess test data
            x_test_sampled = x_test.sample(
                n=n_samples_evaluate, random_state=random_state_evaluate
            )
            y_test_sampled = y_test.loc[x_test_sampled.index]

            x_test_processed = predictor.named_steps["preprocessor"].transform(
                x_test_sampled
            )
            if "classifier" not in predictor.named_steps:
                logger.error("Missing 'classifier' step in the pipeline.")
                raise ValueError("Invalid pipeline: 'classifier' step not found.")

            # Debug: Print shapes of x_test_processed and y_test_sampled
            print(
                f"x_test_processed shape: {x_test_processed.shape}, y_test_sampled shape: {y_test_sampled.shape}"
            )

            # Predict and print evaluation metrics
            y_pred = predictor.named_steps["classifier"].predict(x_test_processed)

            try:
                accuracy = accuracy_score(y_test_sampled, y_pred)
                precision = precision_score(y_test_sampled, y_pred, average="weighted")
                recall = recall_score(y_test_sampled, y_pred, average="weighted")
                f1 = f1_score(y_test_sampled, y_pred, average="weighted")
                conf_matrix = confusion_matrix(y_test_sampled, y_pred)

                print("Accuracy:", accuracy)
                print("Precision:", precision)
                print("Recall:", recall)
                print("F1 Score:", f1)
                print("Confusion Matrix:\n", conf_matrix)
            except ValueError as ve:
                logger.error("ValueError during metric calculation: %s", ve)
                raise
    except ValueError as ve:
        logger.error("ValueError in evaluation: %s", ve)
        raise
    except KeyError as ke:
        logger.error("KeyError in evaluation: %s", ke)
        raise
    except Exception as e:  # Catch-all exception for unexpected errors
        logger.error("Unexpected error in evaluation: %s", e)
        raise
