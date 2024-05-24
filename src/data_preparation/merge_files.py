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


def merge_files(file_prefix: str, num_files: int, output_file: str) -> None:
    """
    Merges CSV files into one big file.

    This function concatenates multiple CSV files that share a common prefix
    into a single output file. It only proceeds with the merge if the output file
    does not already exist, preventing accidental overwriting. The function
    includes robust error handling to catch file-related and unexpected errors,
    logging them for further analysis.

    Args:
        file_prefix (str): The common prefix of the CSV files to be merged.
        num_files (int): The total number of files to merge.
        output_file (str): The name of the output file to create.

    Raises:
        FileNotFoundError: If one or more of the input files cannot be found.
        PermissionError: If there's a permission issue preventing writing to the output file.
        Exception: If an unexpected error occurs during the merge process.
    """
    logger = create_error_logger()
    try:
        if not os.path.exists(output_file):
            with open(output_file, "wb") as out_file:
                for i in range(num_files):
                    try:
                        input_file = f"{file_prefix}_{i}.csv"
                        with open(input_file, "rb") as in_file:
                            out_file.write(in_file.read())
                    except FileNotFoundError:
                        logger.error("File not found: %s", input_file)
                        raise

    except PermissionError:
        logger.error("Permission error while writing to: %s", output_file)
        raise
    except Exception as e:
        logger.error("An unexpected error occurred: %s", e)
        raise
