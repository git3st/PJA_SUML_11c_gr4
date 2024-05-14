"""
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
"""

from autogluon.tabular import TabularDataset


# Model evaluation
def evaluate_model(x_test, y_test, predictor):

    test_data = TabularDataset(x_test.sample(n=100, random_state=0))

    predictions = predictor.predict(test_data)
    print(predictor.leaderboard())
    print(predictions)

    # Export results to data\\04_evaluation_results\\


"""
# Model evaluation (OLD)
def evaluate_model(x_test, y_test, classifier):
    y_pred = classifier.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="binary")
    recall = recall_score(y_test, y_pred, average="binary")
    f1 = f1_score(y_test, y_pred, average="binary")
    conf_matrix = confusion_matrix(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Confusion Matrix:\n", conf_matrix)
"""
