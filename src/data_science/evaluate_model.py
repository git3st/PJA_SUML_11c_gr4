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
import wandb
import matplotlib.pyplot as plt
import seaborn as sns


def create_error_logger() -> logging.Logger:
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
    logger = create_error_logger()
    accuracy = 0
    try:
        if isinstance(predictor, TabularPredictor):
            sample_size = min(n_samples_evaluate, len(x_test))
            test_data = TabularDataset(
                x_test.sample(
                    n=sample_size,
                    random_state=random_state_evaluate,
                    replace=True,
                )
            )
            predictions = predictor.predict(test_data)
            leaderboard = predictor.leaderboard()
            print(leaderboard)
            print(predictions)
            wandb.log({"model_type": "AutoML"})
            accuracy = leaderboard["score_val"][0]  # Assuming the best model is at the top
        else:
            sample_size = min(n_samples_evaluate, len(x_test))
            x_test_sampled = x_test.sample(
                n=sample_size, random_state=random_state_evaluate
            )
            y_test_sampled = y_test.loc[x_test_sampled.index]

            x_test_processed = predictor.named_steps["preprocessor"].transform(
                x_test_sampled
            )
            if "classifier" not in predictor.named_steps:
                logger.error("Missing 'classifier' step in the pipeline.")
                raise ValueError("Invalid pipeline: 'classifier' step not found.")

            y_pred = predictor.named_steps["classifier"].predict(x_test_processed)

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

            wandb.log({
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "model_type": "ML Pipeline"
            })

            fig, ax = plt.subplots()
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            ax.set_title('Confusion Matrix')
            wandb.log({"confusion_matrix": wandb.Image(fig)})
            plt.close(fig)
    except ValueError as ve:
        logger.error("ValueError in evaluation: %s", ve)
        raise
    except KeyError as ke:
        logger.error("KeyError in evaluation: %s", ke)
        raise
    except Exception as e:
        logger.error("Error in evaluation: %s", e)
        raise

    return accuracy, predictor
