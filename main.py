from data_preprocessing import preprocess_data
from merge_files import merge_files


# Run once for full dataset local copy
# file_prefix = "games_metadata_profile_2024_01"
# num_files = 16
# output_file = "full_dataset.csv"
# merge_files(file_prefix, num_files, output_file)


train, test, validate = preprocess_data('full_dataset.csv')
