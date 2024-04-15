def merge_files(file_prefix, num_files, output_file):
    with open(output_file, 'wb') as out_file:
        for i in range(num_files):
            input_file = f"{file_prefix}_{i}.csv"
            with open(input_file, 'rb') as in_file:
                out_file.write(in_file.read())

