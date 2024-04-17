# Machine Learning - Random Forest
from sklearn.ensemble import RandomForestClassifier


def machine_learning(train, test, target_column):

    x_train = train.drop(target_column, axis=1)
    y_train = train[target_column]

    x_test = test.drop(target_column, axis=1)
    y_test = test[target_column]

    # Encoding categorical features "Type 1" and "Type 2"
    categorical_features = ["Event"]
    numeric_features = x_train.select_dtypes(include=["int64", "float64"]).columns

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)

    return model, x_test, y_test
