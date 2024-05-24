import argparse
from data_preparation.merge_files import merge_files
from data_preparation.data_preprocessing import preprocess_data
from data_science.machine_learning import machine_learning
from data_science.evaluate_model import evaluate_model
from data_science.release_model import release_model
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Chess Game Result Predictor")
    parser.add_argument(
        "--file_prefix",
        type=str,
        default="data\\01_raw_data\\games_metadata_profile_2024_01",
    )
    parser.add_argument("--num_files", type=int, default=16)
    parser.add_argument(
        "--output_file", type=str, default="data\\01_raw_data\\full_dataset.csv"
    )
    parser.add_argument("--use_automl", action="store_true", default=False)
    parser.add_argument(
        "--train",
        type=float,
        default=0.8,
        help="Proportion of the dataset to include in the train split.",
    )
    parser.add_argument(
        "--test",
        type=float,
        default=0.10,
        help="Proportion of the dataset to include in the test split.",
    )
    parser.add_argument(
        "--validation",
        type=float,
        default=0.10,
        help="Proportion of the dataset to include in the validation split.",
    )
    parser.add_argument(
        "--seed", type=int, default=50, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=500,
        help="Number of samples to use for training AutoML.",
    )
    parser.add_argument(
        "--time_limit",
        type=int,
        default=60,
        help="Time limit for AutoML training in seconds.",
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=100,
        help="Number of trees in the random forest.",
    )
    parser.add_argument(
        "--n_estimators_pipeline",
        type=int,
        default=100,
        help="Number of trees in the random forest for pipeline.",
    )
    parser.add_argument(
        "--random_state_pipeline",
        type=int,
        default=42,
        help="Random state for the pipeline's random forest.",
    )
    parser.add_argument(
        "--n_samples_evaluate",
        type=int,
        default=100,
        help="Number of samples to use for evaluation.",
    )
    parser.add_argument(
        "--random_state_evaluate",
        type=int,
        default=0,
        help="Random state for evaluation samples.",
    )

    args = parser.parse_args()

    merge_files(args.file_prefix, args.num_files, args.output_file)

    x_train, y_train, x_test, y_test, validate_set, pipeline = preprocess_data(
        filename=args.output_file,
        train=args.train,
        test=args.test,
        validation=args.validation,
        seed=args.seed,
        n_estimators_pipeline=args.n_estimators_pipeline,
        random_state_pipeline=args.random_state_pipeline,
    )

    model = machine_learning(
        x_train,
        y_train,
        validate_set,
        args.use_automl,
        args.n_samples,
        args.time_limit,
        args.n_estimators,
        args.seed,
        pipeline,
    )
    evaluate_model(
        x_test, y_test, model, args.n_samples_evaluate, args.random_state_evaluate
    )
    # release_model()


if __name__ == "__main__":
    main()
