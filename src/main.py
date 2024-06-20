import argparse
import os
from data_preparation.merge_files import merge_files
from data_preparation.data_preprocessing import transform_data
from data_science.machine_learning import machine_learning
from data_science.evaluate_model import evaluate_model
from deployment.release_model import release_model
import wandb

def setup_wandb(project_name: str, api_key: str):
    wandb.login(key=api_key)
    wandb.init(project=project_name)

def main():
    current_path = os.path.realpath(__file__)
    parent_dir = os.path.dirname(os.path.dirname(current_path))

    parser = argparse.ArgumentParser(description="Chess Game Result Predictor")
    parser.add_argument(
        "--file_prefix",
        type=str,
        default=os.path.join(
            parent_dir,
            "data",
            "01_raw_data",
            "games_metadata_profile_2024_01",
        ),
    )
    parser.add_argument("--num_files", type=int, default=16)
    parser.add_argument(
        "--output_file",
        type=str,
        default=os.path.join(parent_dir, "data", "01_raw_data", "full_dataset.csv"),
    )
    parser.add_argument("--use_automl", action="store_true", default=True)
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
    parser.add_argument(
        "--wandb_project",
        type=str,
        help="WandB project name",
    )
    parser.add_argument(
        "--wandb_api_key",
        type=str,
        help="WandB API key",
    )

    args = parser.parse_args()

    wandb.login(key=args.wandb_api_key)
    wandb.init(project=args.wandb_project)

    merge_files(args.file_prefix, args.num_files, args.output_file)

    x_train, y_train, x_test, y_test, validate_set, pipeline = transform_data(
        filename=args.output_file,
        train=args.train,
        test=args.test,
        validation=args.validation,
        seed=args.seed,
        n_estimators_pipeline=args.n_estimators_pipeline,
        random_state_pipeline=args.random_state_pipeline,
    )

    # Train and evaluate AutoML model
    model_automl = machine_learning(
        x_train,
        y_train,
        validate_set,
        use_automl=True,
        n_samples=args.n_samples,
        time_limit=args.time_limit,
        n_estimators=args.n_estimators,
        random_state=args.seed,
        pipeline=pipeline
    )
    accuracy_automl, _ = evaluate_model(
        x_test, y_test, model_automl, args.n_samples_evaluate, args.random_state_evaluate
    )

    # Train and evaluate ML pipeline model
    model_ml = machine_learning(
        x_train,
        y_train,
        validate_set,
        use_automl=False,
        n_samples=args.n_samples,
        time_limit=args.time_limit,
        n_estimators=args.n_estimators,
        random_state=args.seed,
        pipeline=pipeline
    )
    accuracy_ml, _ = evaluate_model(
        x_test, y_test, model_ml, args.n_samples_evaluate, args.random_state_evaluate
    )

    # Compare and select the best model
    if accuracy_automl > accuracy_ml:
        best_model = model_automl
        best_model_type = "AutoML"
    else:
        best_model = model_ml
        best_model_type = "ML Pipeline"

    wandb.log({"best_model_type": best_model_type})
    
    release_model(best_model)

if __name__ == "__main__":
    main()
