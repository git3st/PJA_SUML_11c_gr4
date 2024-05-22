from autogluon.tabular import TabularDataset, TabularPredictor


def machine_learning(
    x_train,
    y_train,
    validate_set,
    use_automl,
    n_samples,
    time_limit,
    n_estimators,
    random_state,
    pipeline,
):
    if use_automl:
        # Merge with x_train and y_train
        train_data = x_train.sample(n=n_samples, random_state=random_state)
        train_data["Result"] = y_train.loc[train_data.index]

        # Debug: Print columns in train_data
        print("Columns in train_data for AutoML:", train_data.columns)
        predictor = TabularPredictor(
            label="Result", path="models", eval_metric="accuracy"
        ).fit(train_data, time_limit=time_limit, presets="medium_quality")
        return predictor
    else:
        # Adjust to training data
        pipeline.fit(x_train, y_train)
        x_train_processed = pipeline.named_steps["preprocessor"].transform(x_train)
        model = pipeline.named_steps["classifier"]
        model.fit(x_train_processed, y_train)
        return pipeline
