from autogluon.tabular import TabularDataset, TabularPredictor

def machine_learning(x_train, y_train, validate_set, use_automl, n_samples, time_limit, n_estimators, random_state):
    if use_automl:
        train_data = TabularDataset(x_train.sample(n=n_samples, random_state=random_state))
        predictor = TabularPredictor(
            label="Result", path="models", eval_metric="accuracy"
        ).fit(train_data, time_limit=time_limit, presets="medium_quality")
        return predictor
    else:
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        model.fit(x_train, y_train)
        return model
