import pickle
from sklearn.base import is_classifier
from data_preparation.data_preprocessing import transform_data
import logging
import os

def create_error_logger() -> logging.Logger:
    """
    Creates a logger to record errors during pipeline execution.

    Returns:
        logging.Logger: The configured logger object.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.ERROR)
    return logger

def release_model(model):
    """
    Saves the trained model to a pickle file.

    This function serializes the model objects into a pickle file for later use. It handles potential errors
    during the file writing and pickling process.

    Args:
        model: The trained model to be saved.

    Raises:
        FileNotFoundError: If the specified file or directory path is not found.
        PermissionError: If the script lacks write permission to the file or directory.
        pickle.PicklingError: If there's an issue pickling the model objects.
        Exception: For any other unexpected errors during model release.
    """
    logger = create_error_logger()
    try:
        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        )
        output_path = os.path.join(
            project_root, "models", "models", "chess_game_result_classifier.pkl"
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(model, f)
    except (FileNotFoundError, PermissionError) as e:
        logger.error("Error opening file for writing: %s", e)
    except pickle.PicklingError as e:
        logger.error("Error pickling model objects: %s", e)
    except Exception as e:
        logger.error("Unexpected error during model release: %s", e)

