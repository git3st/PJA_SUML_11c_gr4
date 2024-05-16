from autogluon.tabular import TabularDataset

def evaluate_model(x_test, y_test, predictor, n_samples_evaluate, random_state_evaluate):
    test_data = TabularDataset(x_test.sample(n=n_samples_evaluate, random_state=random_state_evaluate))
    predictions = predictor.predict(test_data)
    print(predictor.leaderboard())
    print(predictions)
