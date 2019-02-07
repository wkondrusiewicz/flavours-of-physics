n = 5  # cv-folds
n_estimators_forest = 15
n_estimators_xgb = 100
learning_rate = 0.05
max_depth = 4
subsample = 0.5
n_jobs = 4
early_stopping_rounds = 10
min_child_weight = 15
epochs = 8

train_params_nn = {'batch_size': 64, 'epochs': epochs, 'verbose': 0}
train_params_xgb = {
    'early_stopping_rounds': early_stopping_rounds, 'verbose': 0}
