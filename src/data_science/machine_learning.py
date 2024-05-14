from autogluon.tabular import TabularDataset, TabularPredictor


def machine_learning(x_train, y_train, validate_set, pipeline):

    # pipeline.fit(x_train, y_train)
    # return pipeline

    train_data = TabularDataset(x_train.sample(n=500, random_state=0))

    predictor = TabularPredictor(
        label="Result", path="models", eval_metric="accuracy"
    ).fit(train_data, time_limit=60, presets="medium_quality")

    return predictor
