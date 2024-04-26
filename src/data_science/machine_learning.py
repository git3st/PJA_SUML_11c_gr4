# Machine Learning - Random Forest
from sklearn.ensemble import RandomForestClassifier


def machine_learning(train, validate):

    # Encoding features
    categorical_features = ["Event"]  # TO DO
    numeric_features = train.select_dtypes(include=["int64", "float64"]).columns

    # TO DO
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(train, validate)

    return model
