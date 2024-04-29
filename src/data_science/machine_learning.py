# Machine Learning - Random Forest
from sklearn.ensemble import RandomForestClassifier


def machine_learning(train, validate):

    X_train = train.drop("Result", axis=1)
    y_train = train["Result"]

    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X_train, y_train)

    return classifier
