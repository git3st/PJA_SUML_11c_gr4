import logging
from typing import Union
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import pandas as pd
from sklearn.pipeline import Pipeline
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def create_error_logger() -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.ERROR)
    return logger


def log_classification_report_to_wandb(report, prefix="classification_report"):
    for key, value in report.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                wandb.log({f"{prefix}.{key}.{sub_key}": sub_value})
        else:
            wandb.log({f"{prefix}.{key}": value})


def plot_metrics(metrics, model_type):
    steps = np.arange(len(metrics["accuracy"]))
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(steps, metrics["accuracy"], label="Accuracy", marker="o")
    plt.plot(steps, metrics["precision"], label="Precision", marker="o")
    plt.plot(steps, metrics["recall"], label="Recall", marker="o")
    plt.plot(steps, metrics["f1_score"], label="F1 Score", marker="o")
    plt.xlabel("Steps")
    plt.ylabel("Scores")
    plt.title(f"{model_type} Model Metrics")
    plt.legend()

    plt.subplot(1, 2, 2)
    sns.heatmap(metrics["confusion_matrix"][-1], annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{model_type} Confusion Matrix")

    wandb.log({f"{model_type}_metrics_plot": wandb.Image(plt)})
    plt.close()


def evaluate_model(
    x_test: pd.DataFrame,
    y_test: pd.Series,
    predictor: Union[TabularPredictor, Pipeline],
    n_samples_evaluate: int,
    random_state_evaluate: int,
    model_type: str,
):
    logger = create_error_logger()
    accuracy = []
    precision = []
    recall = []
    f1_score_list = []
    confusion_matrices = []
    metrics = {}
    try:
        sample_size = min(n_samples_evaluate, len(x_test))
        x_test_sampled = x_test.sample(
            n=sample_size, random_state=random_state_evaluate
        )
        y_test_sampled = y_test.loc[x_test_sampled.index]

        if isinstance(predictor, TabularPredictor):
            test_data = TabularDataset(x_test_sampled)
            predictions = predictor.predict(test_data)
            accuracy.append(accuracy_score(y_test_sampled, predictions))
            precision.append(
                precision_score(y_test_sampled, predictions, average="weighted")
            )
            recall.append(recall_score(y_test_sampled, predictions, average="weighted"))
            f1_score_list.append(
                f1_score(y_test_sampled, predictions, average="weighted")
            )
            confusion_matrices.append(confusion_matrix(y_test_sampled, predictions))
            class_report = classification_report(
                y_test_sampled, predictions, output_dict=True
            )

            metrics = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score_list,
                "confusion_matrix": confusion_matrices,
            }

            wandb.log({f"{model_type}_accuracy": accuracy[-1]})
            wandb.log({f"{model_type}_precision": precision[-1]})
            wandb.log({f"{model_type}_recall": recall[-1]})
            wandb.log({f"{model_type}_f1_score": f1_score_list[-1]})
            log_classification_report_to_wandb(
                class_report, prefix=f"{model_type}_classification_report"
            )

        else:
            x_test_processed = predictor.named_steps["preprocessor"].transform(
                x_test_sampled
            )
            if "classifier" not in predictor.named_steps:
                logger.error("Missing 'classifier' step in the pipeline.")
                raise ValueError("Invalid pipeline: 'classifier' step not found.")

            y_pred = predictor.named_steps["classifier"].predict(x_test_processed)

            accuracy.append(accuracy_score(y_test_sampled, y_pred))
            precision.append(
                precision_score(y_test_sampled, y_pred, average="weighted")
            )
            recall.append(recall_score(y_test_sampled, y_pred, average="weighted"))
            f1_score_list.append(f1_score(y_test_sampled, y_pred, average="weighted"))
            confusion_matrices.append(confusion_matrix(y_test_sampled, y_pred))
            class_report = classification_report(
                y_test_sampled, y_pred, output_dict=True
            )

            print("Accuracy:", accuracy[-1])
            print("Precision:", precision[-1])
            print("Recall:", recall[-1])
            print("F1 Score:", f1_score_list[-1])
            print("Confusion Matrix:\n", confusion_matrices[-1])

            metrics = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score_list,
                "confusion_matrix": confusion_matrices,
            }

            wandb.log({f"{model_type}_accuracy": accuracy[-1]})
            wandb.log({f"{model_type}_precision": precision[-1]})
            wandb.log({f"{model_type}_recall": recall[-1]})
            wandb.log({f"{model_type}_f1_score": f1_score_list[-1]})
            log_classification_report_to_wandb(
                class_report, prefix=f"{model_type}_classification_report"
            )

        plot_metrics(metrics, model_type)

    except ValueError as ve:
        logger.error("ValueError in evaluation: %s", ve)
        raise
    except KeyError as ke:
        logger.error("KeyError in evaluation: %s", ke)
        raise
    except Exception as e:
        logger.error("Error in evaluation: %s", e)
        raise

    return accuracy[-1], predictor
