def machine_learning(x_train, y_train, validate_set, pipeline):

    pipeline.fit(x_train, y_train)

    return pipeline
