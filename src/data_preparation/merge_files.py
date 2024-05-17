import os

def merge_files(file_prefix, num_files, output_file):
    """
    Merges csv files into one big file, only if the output file doesn't exist in the provided location.

    Args:
        file_prefix: The prefix of the files to merge.
        num_files: The number of files to merge.
        output_file: The name of the output file.
    """

    if not os.path.exists(output_file):
        with open(output_file, "wb") as out_file:
            for i in range(num_files):
                input_file = f"{file_prefix}_{i}.csv"
                with open(input_file, "rb") as in_file:
                    out_file.write(in_file.read())
