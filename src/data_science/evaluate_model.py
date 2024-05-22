from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def evaluate_model(
    x_test, y_test, predictor, n_samples_evaluate, random_state_evaluate
):
    """
    Evaluates a machine learning model on a given test dataset.

    This function handles evaluation differently based on whether the predictor
    is an AutoGluon `TabularPredictor` or a Scikit-learn pipeline with a named
    preprocessor step. It calculates common classification metrics and prints the
    results.

    Args:
        x_test (pd.DataFrame or scipy.sparse.csr_matrix): Features for the test set.
        y_test (pd.Series): Target variable for the test set.
        predictor: A trained machine learning model, either a TabularPredictor or
                   a Scikit-learn Pipeline.
        n_samples_evaluate (int): Number of samples to use for evaluation.
        random_state_evaluate (int): Random seed for sampling the test data.

    Returns:
        None: The function prints evaluation metrics but does not return a value.
    """
    if isinstance(predictor, TabularPredictor):
        # AutoGluon TabularPredictor
        test_data = TabularDataset(
            x_test.sample(n=n_samples_evaluate, random_state=random_state_evaluate)
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

        # Debug: Print shapes of x_test_processed and y_test_sampled
        print(
            f"x_test_processed shape: {x_test_processed.shape}, y_test_sampled shape: {y_test_sampled.shape}"
        )

        # Predict and print evaluation metrics
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
